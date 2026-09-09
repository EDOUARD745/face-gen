"""Évaluation PAR GROUPE DÉMOGRAPHIQUE — l'argument éthique quantifié.

Une moyenne globale peut masquer un effondrement de qualité sur un groupe
minoritaire. Ce script mesure, pour chacun des 7 groupes de peau
(et chaque genre) :
  * FID du groupe : visages générés du groupe vs visages réels du groupe
  * fidélité aux attributs restreinte au groupe

Produit figures/group_eval.csv + figures/group_eval.png (barres).

Usage :
    python -m src.eval_by_group --ckpt runs/ddpm/ckpt_last.pt \
        --clf runs/classifier/attr_clf.pt --data-root data/fairface -n 1000
"""
import argparse
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from .config import Config
from .utils import get_device
from .data import make_dataset, denormalize_age, SKIN_GROUPS
from .evaluate import load_generator, to_uint8
from .train_classifier import AttributeClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("-n", "--num-samples", type=int, default=1000,
                   help="visages générés PAR groupe")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=3.0)
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

    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size
    real_ds = make_dataset(cfg.data, split="train", train_tf=False)

    # Index des images réelles par groupe de peau
    if hasattr(real_ds, "df"):
        groups_real = {s: real_ds.df.index[real_ds.df["race"] == g].tolist()
                       for s, g in enumerate(SKIN_GROUPS)}
    else:  # UTKFace
        groups_real = {s: [i for i, (_, _, _, r) in enumerate(real_ds.files)]
                       for s in range(len(SKIN_GROUPS))}

    rows = []
    for s, gname in enumerate(SKIN_GROUPS):
        idx = groups_real.get(s, [])
        if len(idx) < 100:
            print(f"[skip] {gname} : trop peu d'images réelles ({len(idx)})")
            continue
        print(f"\n=== Groupe : {gname} ({len(idx)} images réelles) ===")
        fid = FrechetInceptionDistance(feature=2048).to(metric_device)

        sub = torch.utils.data.Subset(real_ds, idx[:max(args.num_samples, 1000)])
        for imgs, _ in DataLoader(sub, batch_size=args.batch_size, num_workers=4):
            fid.update(to_uint8(imgs).to(metric_device), real=True)

        correct_g = 0
        age_mae = 0.0
        done = 0
        while done < args.num_samples:
            b = min(args.batch_size, args.num_samples - done)
            attrs = {
                "age": torch.tensor(rng.uniform(0, 1, b), dtype=torch.float32,
                                    device=device),
                "gender": torch.tensor(rng.integers(0, 2, b), device=device),
                "skin": torch.full((b,), s, dtype=torch.long, device=device),
            }
            x = gen(attrs, b, guidance_scale=args.guidance_scale)
            fid.update(to_uint8(x).to(metric_device), real=False)
            with torch.no_grad():
                age_p, gender_p, skin_p = clf(x)
            correct_g += (skin_p.argmax(1) == s).sum().item()
            age_mae += (denormalize_age(age_p)
                        - denormalize_age(attrs["age"])).abs().sum().item()
            done += b

        row = dict(group=gname, n_real=len(idx),
                   fid=fid.compute().item(),
                   acc_skin=100 * correct_g / args.num_samples,
                   age_mae=age_mae / args.num_samples)
        rows.append(row)
        print("  ->", {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in row.items()})

    # ---------------- CSV + figure ----------------
    import csv
    with open(out / "group_eval.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["group"].replace("_", " ") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(names, [r["fid"] for r in rows], color="#7c6cff")
    axes[0].set_title("FID par groupe ↓ (équité de qualité)")
    axes[1].bar(names, [r["acc_skin"] for r in rows], color="#00a58c")
    axes[1].set_title("Fidélité au groupe demandé ↑ (%)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Équité démographique du générateur", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "group_eval.png", dpi=180)
    print(f"\nFigure -> {out/'group_eval.png'}\nDonnées -> {out/'group_eval.csv'}")


if __name__ == "__main__":
    main()
