"""Diagnostic de la réponse en âge + calibration post-hoc de la condition.

Motivation (voir rapport, section « limites ») : les labels d'âge de FairFace
sont des tranches, converties en points médians. Après le filtre 18-70 ans du
sujet, le support d'entraînement se réduit à cinq valeurs discrètes
(24,5 / 34,5 / 44,5 / 54,5 / 64,5 ans) : le modèle n'a JAMAIS vu d'exemple
sous 24,5 ans ni au-dessus de 64,5 ans. Toute demande hors de cet intervalle
est une extrapolation, ce qui gonfle mécaniquement la MAE d'âge mesurée sur
un protocole uniforme 18-70.

Ce script mesure la fonction de réponse âge_réalisé = f(âge_demandé) avec le
classifieur d'attributs comme juge, puis en déduit une CALIBRATION post-hoc :
f étant monotone mais compressée, on inverse f pour choisir, à l'inférence, la
valeur de condition qui produit réellement l'âge voulu. Aucune ré-estimation
des poids du modèle n'est nécessaire.

Sorties :
  * figures/age_response.png / .csv  -- courbe de réponse (figure du rapport)
  * runs/ddpm/age_calibration.json   -- table de calibration réutilisable

Usage :
    python -m src.age_response --ckpt runs/ddpm/ckpt_last.pt \
        --clf runs/classifier/attr_clf.pt --image-size 48 -n 48
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from .data import SKIN_GROUPS, denormalize_age, normalize_age, FAIRFACE_AGE_MIDPOINTS
from .train_classifier import AttributeClassifier
from .utils import config_from_ckpt, get_device, load_ddpm_from_ckpt


def training_support(min_age=18, max_age=70, age_jitter=False):
    """Valeurs d'âge réellement présentes dans FairFace après le filtre.

    Avec `age_jitter`, l'âge est tiré dans la tranche annotée : le support est
    l'intervalle continu [min_age, max_age] et non cinq points médians.
    """
    if age_jitter:
        return [float(min_age), float(max_age)]
    return sorted(v for v in FAIRFACE_AGE_MIDPOINTS.values()
                  if min_age <= v <= max_age)


@torch.no_grad()
def measure_response(diff, clf, image_size, device, ages, n_per_age,
                     guidance_scale, batch_size, seed=0):
    """Pour chaque âge demandé, renvoie la distribution des âges perçus."""
    rng = np.random.default_rng(seed)
    rows = []
    for age in ages:
        preds = []
        done = 0
        while done < n_per_age:
            b = min(batch_size, n_per_age - done)
            attrs = {
                "age": torch.full((b,), normalize_age(age), device=device),
                "gender": torch.tensor(rng.integers(0, 2, b), device=device),
                "skin": torch.tensor(rng.integers(0, len(SKIN_GROUPS), b),
                                     device=device),
            }
            x = diff.sample_ddim(attrs, (b, 3, image_size, image_size),
                                 steps=50, guidance_scale=guidance_scale)
            age_p, _, _ = clf(x)
            preds.append(denormalize_age(age_p).cpu().numpy())
            done += b
        preds = np.concatenate(preds)
        rows.append({"age_demande": float(age),
                     "age_percu_moyen": float(preds.mean()),
                     "age_percu_std": float(preds.std()),
                     "mae": float(np.abs(preds - age).mean())})
        print(f"  demandé {age:5.1f} -> perçu {preds.mean():5.1f} "
              f"± {preds.std():4.1f}  (MAE {rows[-1]['mae']:5.1f})")
    return rows


def build_calibration(rows, min_age=18, max_age=70):
    """Inverse la fonction de réponse : âge voulu -> valeur de condition.

    La réponse mesurée est bruitée ; on la rend monotone croissante
    (régression isotone « pauvre » par maximum cumulé) avant inversion,
    sinon l'interpolation inverse n'est pas définie.
    """
    req = np.array([r["age_demande"] for r in rows])
    got = np.array([r["age_percu_moyen"] for r in rows])
    got_mono = np.maximum.accumulate(got)
    # petite pente minimale pour garder l'inversion bien conditionnée
    for i in range(1, len(got_mono)):
        got_mono[i] = max(got_mono[i], got_mono[i - 1] + 1e-3)

    targets = np.linspace(min_age, max_age, 27)
    conds = np.interp(targets, got_mono, req)          # inversion
    conds = np.clip(conds, min_age, max_age)
    # Une calibration n'a de sens que si la réponse VARIE : si le modèle rend
    # le même âge quelle que soit la demande, la fonction n'est pas inversible
    # et la « corriger » ne ferait qu'habiller un contrôle inexistant.
    amplitude = float(got.max() - got.min())
    applicable = amplitude >= 10.0
    if not applicable:
        print(f"\n⚠ Amplitude de réponse = {amplitude:.1f} ans (< 10) : la "
              f"condition d'âge n'est pas suivie, calibration NON applicable "
              f"(marquée comme telle dans le JSON).")
    return {"applicable": applicable,
            "amplitude_reponse_annees": amplitude,
            "cible_annees": targets.tolist(),
            "condition_annees": conds.tolist(),
            "reponse_demande": req.tolist(),
            "reponse_percue": got.tolist()}


def plot(rows, support, out_png, continu=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    req = [r["age_demande"] for r in rows]
    got = [r["age_percu_moyen"] for r in rows]
    std = [r["age_percu_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    lo, hi = min(support), max(support)
    ax.axvspan(lo, hi, color="#dbeafe", zorder=0,
               label=f"support d'entraînement ({lo:.1f}-{hi:.1f} ans"
                     f"{', continu' if continu else f', {len(support)} valeurs'})")
    if not continu:
        for s in support:
            ax.axvline(s, color="#93c5fd", lw=1, ls=":", zorder=1)
    ax.plot([18, 70], [18, 70], "k--", lw=1, label="contrôle parfait")
    ax.errorbar(req, got, yerr=std, marker="o", color="#b91c1c", capsize=3,
                lw=1.8, label="réponse mesurée (juge = classifieur)")
    ax.set_xlabel("âge demandé (années)")
    ax.set_ylabel("âge perçu sur le visage généré (années)")
    ax.set_title("Réponse du DDPM conditionnel à la condition d'âge")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    print(f"figure -> {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ddpm/ckpt_last.pt")
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("-n", "--per-age", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--guidance-age", type=float, default=None,
                   help="échelle de guidage propre à l'âge : active le guidage "
                        "compositionnel (modèle entraîné avec "
                        "--independent-drop requis)")
    p.add_argument("--guidance-mode", choices=["hybrid", "compositional"],
                   default="hybrid",
                   help="hybrid : conditionnement conjoint + supplément d'âge "
                        "(défaut) ; compositional : somme de branches "
                        "mono-attribut (contrôle indépendant, coût en réalisme)")
    p.add_argument("--step", type=float, default=4.0, help="pas de la grille d'âges")
    p.add_argument("--out-json", default="runs/ddpm/age_calibration.json")
    p.add_argument("--out-png", default="figures/age_response.png")
    p.add_argument("--out-csv", default="figures/age_response.csv")
    args = p.parse_args()

    device = get_device()
    ck_cfg = config_from_ckpt(torch.load(args.ckpt, map_location="cpu"))
    jitter = getattr(ck_cfg.data, "age_jitter", False)
    _, diff, image_size = load_ddpm_from_ckpt(args.ckpt, device, args.image_size)
    clf = AttributeClassifier().to(device)
    clf.load_state_dict(torch.load(args.clf, map_location=device))
    clf.eval()

    support = training_support(age_jitter=jitter)
    print(f"Support d'âge vu à l'entraînement : "
          f"{'continu ' + str(support) if jitter else support}")
    scale = args.guidance_scale
    if args.guidance_age is not None:
        if args.guidance_mode == "hybrid":
            # branche conjointe conservée + supplément d'âge
            scale = {"base": args.guidance_scale, "age": args.guidance_age}
        else:
            scale = {"age": args.guidance_age, "gender": args.guidance_scale,
                     "skin": args.guidance_scale}
        print(f"Guidage compositionnel : {scale}")

    ages = np.arange(18, 70 + 1e-6, args.step)
    print(f"Mesure de la réponse sur {len(ages)} âges × {args.per_age} visages "
          f"(w={scale}) …")
    rows = measure_response(diff, clf, image_size, device, ages, args.per_age,
                            scale, args.batch_size)

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("age_demande,age_percu_moyen,age_percu_std,mae\n")
        for r in rows:
            f.write(f"{r['age_demande']},{r['age_percu_moyen']:.3f},"
                    f"{r['age_percu_std']:.3f},{r['mae']:.3f}\n")
    print(f"csv -> {args.out_csv}")

    calib = build_calibration(rows)
    calib["guidance_scale"] = scale if isinstance(scale, dict) else args.guidance_scale
    calib["support_entrainement"] = support
    calib["support_continu"] = jitter
    pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_json).write_text(json.dumps(calib, indent=2))
    print(f"calibration -> {args.out_json}")

    plot(rows, support, args.out_png, continu=jitter)

    mae_all = float(np.mean([r["mae"] for r in rows]))
    in_sup = [r for r in rows if min(support) <= r["age_demande"] <= max(support)]
    mae_sup = float(np.mean([r["mae"] for r in in_sup]))
    print(f"\nMAE moyenne sur 18-70 (protocole uniforme)      : {mae_all:.1f} ans")
    print(f"MAE moyenne dans le support d'entraînement       : {mae_sup:.1f} ans")


if __name__ == "__main__":
    main()
