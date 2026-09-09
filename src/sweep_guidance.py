"""Balayage du poids de guidance w : matérialise le compromis du CFG.

Pour chaque w, mesure :
  * FID            (réalisme distributionnel)
  * fidélité       (accord genre/peau %, MAE âge via classifieur)
  * diversité LPIPS intra-condition

Produit figures/cfg_sweep.png (3 panneaux, prête pour le rapport)
et figures/cfg_sweep.csv (données brutes).

Usage :
    python -m src.sweep_guidance --ckpt runs/ddpm/ckpt_last.pt \
        --clf runs/classifier/attr_clf.pt --data-root data/fairface \
        --scales 1 2 3 5 8 -n 2000
"""
import argparse
import itertools
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from .config import Config
from .utils import get_device
from .data import make_dataset, denormalize_age
from .evaluate import load_generator, random_attrs, to_uint8
from .train_classifier import AttributeClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--scales", type=float, nargs="+", default=[1, 2, 3, 5, 8])
    p.add_argument("-n", "--num-samples", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    device = get_device()
    metric_device = "cpu" if str(device) == "mps" else device  # FID en float64 -> CPU sur MPS
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    gen = load_generator("ddpm", args.ckpt, args.image_size, device)
    clf = AttributeClassifier().to(device)
    clf.load_state_dict(torch.load(args.clf, map_location=device))
    clf.eval()

    import lpips
    lp = lpips.LPIPS(net="alex").to(device)

    # Statistiques réelles calculées une seule fois
    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size
    real_ds = make_dataset(cfg.data, split="train", train_tf=False)
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=4)
    real_batches = []
    n_real = 0
    for imgs, _ in real_dl:
        real_batches.append(to_uint8(imgs))
        n_real += imgs.shape[0]
        if n_real >= args.num_samples:
            break

    rows = []
    for w in args.scales:
        print(f"\n=== guidance w = {w} ===")
        fid = FrechetInceptionDistance(feature=2048).to(metric_device)
        for rb in real_batches:
            fid.update(rb.to(metric_device), real=True)

        correct_g = correct_s = 0
        age_mae = 0.0
        done = 0
        while done < args.num_samples:
            b = min(args.batch_size, args.num_samples - done)
            attrs = random_attrs(b, device, rng)
            x = gen(attrs, b, guidance_scale=w)
            fid.update(to_uint8(x).to(metric_device), real=False)
            with torch.no_grad():
                age_p, gender_p, skin_p = clf(x)
            correct_g += (gender_p.argmax(1) == attrs["gender"]).sum().item()
            correct_s += (skin_p.argmax(1) == attrs["skin"]).sum().item()
            age_mae += (denormalize_age(age_p)
                        - denormalize_age(attrs["age"])).abs().sum().item()
            done += b
            print(f"  {done}/{args.num_samples}")

        # Diversité intra-condition
        div = []
        for _ in range(6):
            a = random_attrs(1, device, rng)
            attrs8 = {k: v.repeat(8) for k, v in a.items()}
            x = gen(attrs8, 8, guidance_scale=w)
            with torch.no_grad():
                for i, j in itertools.combinations(range(8), 2):
                    div.append(lp(x[i:i+1], x[j:j+1]).item())

        row = dict(w=w, fid=fid.compute().item(),
                   acc_gender=100 * correct_g / args.num_samples,
                   acc_skin=100 * correct_s / args.num_samples,
                   age_mae=age_mae / args.num_samples,
                   lpips_div=float(np.mean(div)))
        rows.append(row)
        print("  ->", {k: round(v, 2) for k, v in row.items()})

    # ---------------- Sauvegarde CSV + figure ----------------
    import csv
    with open(out / "cfg_sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ws = [r["w"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(ws, [r["fid"] for r in rows], "o-", color="#7c6cff")
    axes[0].set_title("Réalisme : FID ↓")
    axes[1].plot(ws, [r["acc_gender"] for r in rows], "o-", label="Genre (%)")
    axes[1].plot(ws, [r["acc_skin"] for r in rows], "s-", label="Peau (%)")
    ax1b = axes[1].twinx()
    ax1b.plot(ws, [r["age_mae"] for r in rows], "^--", color="gray",
              label="MAE âge (ans)")
    ax1b.set_ylabel("MAE âge (ans)")
    axes[1].set_title("Fidélité aux attributs ↑")
    axes[1].legend(loc="lower right")
    axes[2].plot(ws, [r["lpips_div"] for r in rows], "o-", color="#00a58c")
    axes[2].set_title("Diversité LPIPS intra-condition ↑")
    for ax in axes:
        ax.set_xlabel("Poids de guidance w")
        ax.grid(alpha=0.3)
    fig.suptitle("Compromis fidélité / diversité du Classifier-Free Guidance",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "cfg_sweep.png", dpi=180)
    print(f"\nFigure -> {out/'cfg_sweep.png'}\nDonnées -> {out/'cfg_sweep.csv'}")


if __name__ == "__main__":
    main()
