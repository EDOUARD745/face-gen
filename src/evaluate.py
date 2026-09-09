"""Évaluation quantitative : FID, IS, LPIPS (diversité), fidélité attributs.

Métriques (voir rapport, section 4) :
  * FID   (Heusel et al., 2017)  : distance de Fréchet entre statistiques
          Inception des images réelles et générées. Plus bas = mieux.
  * IS    (Salimans et al., 2016): qualité + diversité via Inception-v3.
  * LPIPS (Zhang et al., 2018)   : ici utilisé comme mesure de DIVERSITÉ
          intra-condition : distance perceptuelle moyenne entre paires de
          visages générés avec les MÊMES attributs. Trop bas = mode collapse.
  * Fidélité attributs : un classifieur ResNet-18 (train_classifier.py)
          prédit les attributs des visages générés ; on mesure l'accord
          avec les attributs demandés.

Usage :
    python -m src.evaluate --model ddpm --ckpt runs/ddpm/ckpt_last.pt \
        --clf runs/classifier/attr_clf.pt --data-root data/fairface -n 5000
"""
import argparse
import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from .config import Config
from .utils import get_device, load_ddpm_from_ckpt
from .data import make_dataset, SKIN_GROUPS, denormalize_age, normalize_age
from .sampling import load_calibration, make_controlled_sampler
from .models.cgan import Generator
from .train_classifier import AttributeClassifier


# Âges réellement présents dans FairFace après le filtre 18-70 ans : le
# dataset n'annote que des TRANCHES, ramenées à leur point médian. Le protocole
# "support" échantillonne les conditions dans cet ensemble, le protocole
# "uniform" tire uniformément sur 18-70 (et demande donc au modèle des âges
# jamais vus). Les deux sont rapportés dans le rapport.
AGE_SUPPORT = (24.5, 34.5, 44.5, 54.5, 64.5)


def random_attrs(n, device, rng, age_sampling="uniform"):
    if age_sampling == "support":
        years = rng.choice(AGE_SUPPORT, n)
        age = np.array([normalize_age(y) for y in years], dtype=np.float32)
    else:
        age = rng.uniform(0, 1, n)
    return {
        "age": torch.tensor(age, dtype=torch.float32, device=device),
        "gender": torch.tensor(rng.integers(0, 2, n), device=device),
        "skin": torch.tensor(rng.integers(0, len(SKIN_GROUPS), n), device=device),
    }


def load_generator(kind, ckpt_path, image_size, device):
    """Retourne fn(attrs, n) -> images dans [-1, 1].

    Pour le DDPM, l'architecture est reconstruite depuis la config embarquée
    dans le checkpoint (compatible préset mac).
    """
    if kind == "ddpm":
        _, diff, image_size = load_ddpm_from_ckpt(ckpt_path, device)

        def gen(attrs, n, guidance_scale=3.0):
            return diff.sample_ddim(attrs, (n, 3, image_size, image_size),
                                    steps=50, guidance_scale=guidance_scale)
        return gen

    if kind == "cgan":
        ck = torch.load(ckpt_path, map_location=device)
        cfg = Config()
        G = Generator(cfg.gan.latent_dim, image_size=image_size,
                      base=cfg.gan.base_channels).to(device)
        G.load_state_dict(ck["G"])
        G.eval()

        @torch.no_grad()
        def gen(attrs, n, guidance_scale=None):
            z = torch.randn(n, cfg.gan.latent_dim, device=device)
            return G(z, attrs)
        return gen

    raise ValueError(kind)


def to_uint8(x):
    """[-1,1] float -> [0,255] uint8 (format torchmetrics)."""
    return ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ddpm", "cgan"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("-n", "--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--age-sampling", choices=["uniform", "support"],
                   default="uniform",
                   help="uniform = 18-70 (inclut des âges hors support "
                        "d'entraînement) ; support = points médians FairFace")
    p.add_argument("--calibration", default=None,
                   help="JSON de calibration d'âge (src.age_response)")
    p.add_argument("--skip-fid", action="store_true",
                   help="n'évalue que la fidélité aux attributs (rapide)")
    p.add_argument("--fidelity-n", type=int, default=1000)
    p.add_argument("--best-of", type=int, default=1,
                   help="k candidats par condition, le plus conforme est gardé")
    args = p.parse_args()

    device = get_device()
    rng = np.random.default_rng(0)
    raw_gen = load_generator(args.model, args.ckpt, args.image_size, device)

    # Classifieur d'attributs : juge de fidélité, et sélecteur si --best-of > 1
    clf = AttributeClassifier().to(device)
    clf.load_state_dict(torch.load(args.clf, map_location=device))
    clf.eval()

    calibration = load_calibration(args.calibration)
    if calibration is not None:
        print(f"Calibration d'âge active : {args.calibration}")
    if args.best_of > 1:
        print(f"Échantillonnage par rejet : best-of-{args.best_of}")

    _base = lambda attrs, n: raw_gen(attrs, n,
                                     guidance_scale=args.guidance_scale)
    sampler = make_controlled_sampler(_base, clf=clf, calibration=calibration,
                                      best_of=args.best_of)

    def gen(attrs, n, guidance_scale=None):
        return sampler(attrs, n)

    # ---------------- FID & IS ----------------
    if args.skip_fid:
        print("(--skip-fid : FID / IS / LPIPS ignorés)")
    else:
        run_distribution_metrics(args, gen, device, random_attrs)

    # ---------------- Fidélité aux attributs ----------------
    n_eval = args.fidelity_n
    correct_g, correct_s, age_mae, done = 0, 0, 0.0, 0
    while done < n_eval:
        b = min(args.batch_size, n_eval - done)
        attrs = random_attrs(b, device, rng, args.age_sampling)
        x = gen(attrs, b)
        with torch.no_grad():
            age_p, gender_p, skin_p = clf(x)
        correct_g += (gender_p.argmax(1) == attrs["gender"]).sum().item()
        correct_s += (skin_p.argmax(1) == attrs["skin"]).sum().item()
        age_mae += (denormalize_age(age_p) - denormalize_age(attrs["age"])) \
            .abs().sum().item()
        done += b

    print("\n--- Fidélité aux attributs (sur visages générés) ---")
    print(f"protocole d'âge : {args.age_sampling}"
          f"{' + calibration' if calibration is not None else ''}"
          f"{f' + best-of-{args.best_of}' if args.best_of > 1 else ''}")
    print(f"Genre  : {100 * correct_g / n_eval:.1f}% d'accord")
    print(f"Peau   : {100 * correct_s / n_eval:.1f}% d'accord")
    print(f"Âge    : MAE = {age_mae / n_eval:.1f} ans")
    return


def run_distribution_metrics(args, gen, device, random_attrs):
    """FID, IS et diversité LPIPS intra-condition."""
    rng = np.random.default_rng(0)
    # torchmetrics FID/IS accumulent en float64 -> incompatibles MPS ; on les
    # garde sur CPU (sampling et classifieur restent sur le device rapide).
    metric_device = "cpu" if str(device) == "mps" else device
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(metric_device)
    inception = InceptionScore(normalize=False).to(metric_device)

    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size
    real_ds = make_dataset(cfg.data, split="train", train_tf=False)
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=4)
    n_real = 0
    for imgs, _ in real_dl:
        fid.update(to_uint8(imgs).to(metric_device), real=True)
        n_real += imgs.shape[0]
        if n_real >= args.num_samples:
            break

    print(f"Génération de {args.num_samples} visages ({args.model})…")
    n_done = 0
    while n_done < args.num_samples:
        b = min(args.batch_size, args.num_samples - n_done)
        attrs = random_attrs(b, device, rng, args.age_sampling)
        x = gen(attrs, b)
        fid.update(to_uint8(x).to(metric_device), real=False)
        inception.update(to_uint8(x).to(metric_device))
        n_done += b
        print(f"  {n_done}/{args.num_samples}")

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    print(f"\nFID  = {fid_score:.2f}")
    print(f"IS   = {is_mean.item():.2f} ± {is_std.item():.2f}")

    # ---------------- Diversité LPIPS ----------------
    import lpips
    loss_lpips = lpips.LPIPS(net="alex").to(device)
    div_scores = []
    for _ in range(10):  # 10 conditions fixes, 8 visages chacune
        a = random_attrs(1, device, rng, args.age_sampling)
        attrs8 = {k: v.repeat(8) for k, v in a.items()}
        x = gen(attrs8, 8)
        with torch.no_grad():
            for i, j in itertools.combinations(range(8), 2):
                div_scores.append(
                    loss_lpips(x[i:i + 1], x[j:j + 1]).item())
    print(f"Diversité LPIPS (intra-condition) = {np.mean(div_scores):.3f} "
          f"(> 0.2 souhaitable, ~0 = mode collapse)")


if __name__ == "__main__":
    main()
