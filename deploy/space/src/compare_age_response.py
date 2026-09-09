"""Figure comparative des fonctions de réponse en âge (avant / après).

Superpose plusieurs courbes produites par `src.age_response` pour montrer
l'effet de chaque correctif : données corrigées, masquage indépendant par
attribut, guidage compositionnel.

Usage :
    python -m src.compare_age_response \
        "v1 (support à 5 valeurs)=figures/age_response.csv" \
        "v2 (données + masquage indépendant)=figures/age_response_v2.csv" \
        "v2 + guidage d'âge w=10=figures/age_response_v2_comp.csv" \
        --out figures/age_response_compare.png
"""
import argparse
import csv
import pathlib

import numpy as np

COLORS = ["#b91c1c", "#ea580c", "#15803d", "#1d4ed8", "#7c3aed"]


def load(path):
    rows = [{k: float(v) for k, v in r.items()}
            for r in csv.DictReader(open(path))]
    return (np.array([r["age_demande"] for r in rows]),
            np.array([r["age_percu_moyen"] for r in rows]),
            np.array([r["mae"] for r in rows]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("series", nargs="+", help='"légende=chemin.csv"')
    p.add_argument("--out", default="figures/age_response_compare.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot([18, 70], [18, 70], "k--", lw=1.2, label="contrôle parfait")

    resume = []
    for i, spec in enumerate(args.series):
        label, path = spec.rsplit("=", 1)  # le libellé peut contenir « w=10 »
        req, got, mae = load(path)
        c = COLORS[i % len(COLORS)]
        ax.plot(req, got, marker="o", color=c, lw=2, label=label)
        ax2.plot(req, mae, marker="o", color=c, lw=2, label=label)
        resume.append((label, got.max() - got.min(), mae.mean()))

    ax.set_xlabel("âge demandé (années)")
    ax.set_ylabel("âge perçu (années)")
    ax.set_title("Fonction de réponse en âge")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax2.set_xlabel("âge demandé (années)")
    ax2.set_ylabel("erreur absolue (années)")
    ax2.set_title("Erreur par âge demandé")
    ax2.axhline(6.5, color="#666", ls=":", lw=1,
                label="erreur du juge sur images réelles")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"figure -> {args.out}\n")
    print(f"{'série':<45} {'amplitude':>10} {'MAE moy.':>10}")
    for label, amp, mae in resume:
        print(f"{label:<45} {amp:>9.1f} ans {mae:>7.1f} ans")


if __name__ == "__main__":
    main()
