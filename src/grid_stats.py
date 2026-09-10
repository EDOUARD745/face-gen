"""Suivi de qualité pendant l'entraînement, sans coût GPU.

Les grilles de contrôle écrites par `train_ddpm` sont échantillonnées avec les
poids EMA : elles constituent un échantillon gratuit de ce que le modèle
produit à un instant donné. Ce script les découpe et mesure deux indicateurs,
comparés à la distribution des images réelles à la même résolution :

  * dérive colorimétrique : écart entre les moyennes des trois canaux.
    Au-dessus du 99e centile du réel, le tirage est vert / rose / bleuté.
  * netteté : énergie du laplacien. Une valeur TRÈS supérieure au réel ne
    signale pas du détail mais de la texture parasite (cas des premières
    grilles après un changement de résolution).

Le FID reste la mesure de référence ; ceci sert à décider en cours de route
s'il faut laisser tourner, baisser le pas d'apprentissage ou reprendre depuis
un checkpoint antérieur — décisions qu'on ne peut pas attendre 2 h à chaque
fois pour prendre.

Usage :
    python -m src.grid_stats runs/ddpm_96 --image-size 96
"""
import argparse
import glob
import pathlib
import re

import numpy as np
from PIL import Image


def cut_cells(path, rows=7, cols=6, pad=2):
    im = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255
    h, w = im.shape[0] // rows, im.shape[1] // cols
    return [im[r * h + pad:(r + 1) * h - pad, c * w + pad:(c + 1) * w - pad]
            for r in range(rows) for c in range(cols)]


def colour_cast(a):
    m = a.reshape(-1, 3).mean(0)
    return float(m.max() - m.min())


def sharpness(a):
    g = a.mean(2)
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                 - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.mean())


def real_reference(data_root, image_size, n=512, seed=0):
    """Échantillon réel FIXE : sans seed, la référence bouge d'un appel à
    l'autre (±0,01 sur la dérive) et de faux écarts apparaissent entre deux
    relevés du même run."""
    import torch
    from torch.utils.data import DataLoader
    from .config import Config
    from .data import make_dataset
    cfg = Config().data
    cfg.root, cfg.image_size = data_root, image_size
    ds = make_dataset(cfg, split="val", train_tf=False)
    g = torch.Generator().manual_seed(seed)
    imgs, _ = next(iter(DataLoader(ds, batch_size=n, shuffle=True, generator=g)))
    real = ((imgs.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).numpy()
    return (np.array([colour_cast(a) for a in real]),
            np.array([sharpness(a) for a in real]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--image-size", type=int, required=True)
    p.add_argument("--last", type=int, default=0, help="n dernières grilles")
    args = p.parse_args()

    rc, rs = real_reference(args.data_root, args.image_size)
    p99 = np.percentile(rc, 99)
    print(f"référence réelle {args.image_size}px : dérive méd "
          f"{np.median(rc):.3f} (p99 {p99:.3f}) | netteté méd {np.median(rs):.4f}\n")
    print(f"{'step':>9} {'dérive':>8} {'hors plage':>12} {'netteté':>9} {'% du réel':>10}")

    grids = sorted(glob.glob(str(pathlib.Path(args.run_dir) / "grid_*.png")))
    for path in grids[-args.last:] if args.last else grids:
        cells = cut_cells(path)
        c = np.array([colour_cast(a) for a in cells])
        s = np.array([sharpness(a) for a in cells])
        bad = int((c > p99).sum())
        step = int(re.search(r"grid_(\d+)", path).group(1))
        print(f"{step:>9} {np.median(c):>8.3f} {bad:>7}/{len(cells):<4} "
              f"{np.median(s):>9.4f} {100 * np.median(s) / np.median(rs):>9.0f} %")


if __name__ == "__main__":
    main()
