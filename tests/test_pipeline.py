"""Tests de non-régression (sans GPU ni checkpoint).

Exécution :  python3 -m tests.test_pipeline
Ils verrouillent les invariants que le diagnostic du contrôle d'âge a mis en
évidence : c'est l'absence de ce type de vérification qui a laissé passer un
support d'âge dégénéré pendant 18 h d'entraînement.
"""
import json
import pathlib
import tempfile

import numpy as np
import torch

from src.config import Config
from src.data import (FAIRFACE_AGE_BINS, FAIRFACE_AGE_MIDPOINTS, denormalize_age,
                      make_dataset, normalize_age)
from src.sampling import AgeCalibration, load_calibration, sample_best_of

DATA_ROOT = "data/fairface"


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'ÉCHEC'}] {name}{' — ' + detail if detail else ''}")
    if not cond:
        raise AssertionError(name)


def test_age_encoding_roundtrip():
    for y in (18, 24.5, 45, 70):
        check(f"normalisation réversible ({y} ans)",
              abs(denormalize_age(normalize_age(y)) - y) < 1e-6)
    check("âge hors plage borné", normalize_age(90) == 1.0)


def test_bins_coherents():
    """Les points médians doivent tomber dans leurs bornes de tranche."""
    for label, mid in FAIRFACE_AGE_MIDPOINTS.items():
        lo, hi = FAIRFACE_AGE_BINS[label]
        check(f"médiane cohérente pour {label}", lo <= mid <= hi, f"{mid} ∈ [{lo},{hi}]")


def test_support_age(n=400):
    """Le mode historique dégénère le support ; --age-jitter le restaure."""
    if not pathlib.Path(DATA_ROOT).exists():
        print("  [skip] dataset absent")
        return
    rng = np.random.default_rng(0)
    cfg = Config().data
    cfg.root, cfg.image_size = DATA_ROOT, 48

    ds = make_dataset(cfg, split="val")
    ages = {round(denormalize_age(ds[i][1]["age"].item()), 2)
            for i in rng.integers(0, len(ds), n)}
    check("support historique dégénéré (≤ 5 valeurs)", len(ages) <= 5,
          f"{sorted(ages)}")

    cfg.age_jitter = True
    ds_j = make_dataset(cfg, split="val")
    ages_j = [denormalize_age(ds_j[i][1]["age"].item())
              for i in rng.integers(0, len(ds_j), n)]
    check("support continu avec --age-jitter", len(set(np.round(ages_j, 2))) > 100,
          f"{len(set(np.round(ages_j, 2)))} valeurs distinctes")
    check("18-19 ans présents", min(ages_j) < 20, f"min = {min(ages_j):.1f} ans")
    check("bornes respectées", 18 <= min(ages_j) and max(ages_j) <= 70,
          f"[{min(ages_j):.1f}, {max(ages_j):.1f}]")
    check("plus d'images qu'en mode historique", len(ds_j) > len(ds),
          f"{len(ds_j)} vs {len(ds)}")


def test_calibration_refuse_reponse_plate():
    """Une réponse plate ne doit JAMAIS produire une calibration utilisable."""
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "calib.json"
        p.write_text(json.dumps({"applicable": False,
                                 "amplitude_reponse_annees": 7.7,
                                 "cible_annees": [18, 70],
                                 "condition_annees": [18, 70]}))
        check("calibration non applicable ignorée", load_calibration(p) is None)

        p.write_text(json.dumps({"applicable": True,
                                 "amplitude_reponse_annees": 45.0,
                                 "cible_annees": [20.0, 60.0],
                                 "condition_annees": [25.0, 55.0]}))
        cal = load_calibration(p)
        check("calibration applicable chargée", isinstance(cal, AgeCalibration))
        check("inversion linéaire correcte", abs(cal.years([40.0])[0] - 40.0) < 1e-6)


def test_best_of_selectionne_le_plus_conforme():
    """Le rejet doit converger vers l'attribut demandé, pas vers du bruit."""
    n = 8
    attrs = {"age": torch.full((n,), normalize_age(30.0)),
             "gender": torch.zeros(n, dtype=torch.long),
             "skin": torch.zeros(n, dtype=torch.long)}

    class FakeClf:
        """Juge factice : l'âge perçu est encodé dans le pixel [0,0,0]."""
        def __call__(self, x):
            age = x[:, 0, 0, 0].clamp(0, 1)
            logits = torch.zeros(x.shape[0], 2)
            skin = torch.zeros(x.shape[0], 7)
            return age, logits, skin

    calls = {"i": 0}

    def gen(a, m):
        # 1er lot : âge perçu 0.9 (loin) ; 2e lot : 0.23 (proche de la consigne)
        calls["i"] += 1
        val = 0.9 if calls["i"] == 1 else normalize_age(30.0)
        return torch.full((m, 3, 8, 8), val)

    x = sample_best_of(gen, attrs, n, FakeClf(), k=2)
    check("best-of retient le candidat conforme",
          abs(x[0, 0, 0, 0].item() - normalize_age(30.0)) < 1e-6)
    check("k lots générés", calls["i"] == 2)


def main():
    for fn in (test_age_encoding_roundtrip, test_bins_coherents, test_support_age,
               test_calibration_refuse_reponse_plate,
               test_best_of_selectionne_le_plus_conforme):
        print(f"\n{fn.__name__}")
        fn()
    print("\nTous les tests passent.")


if __name__ == "__main__":
    main()
