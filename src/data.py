"""Chargement des données et encodage des attributs.

Datasets supportés :
  * FairFace (suggéré par le sujet) : ~108k visages, annotés âge / genre /
    "race" (7 groupes), équilibré démographiquement -> limite les biais.
    https://github.com/joojs/fairface
    Structure attendue :
        data/fairface/
            fairface_label_train.csv
            fairface_label_val.csv
            train/xxxxx.jpg
            val/xxxxx.jpg
  * UTKFace : ~20k visages, âge exact + genre + ethnie dans le nom de fichier
    [age]_[gender]_[race]_[date].jpg
    https://susanqq.github.io/UTKFace/

Encodage des attributs (voir rapport, section "Module de contrôle") :
  * âge   -> scalaire continu normalisé dans [0, 1] (contrôle continu 18-70)
  * genre -> classe {0: Homme, 1: Femme}
  * peau  -> classe parmi SKIN_GROUPS (7 groupes FairFace)
"""
import pathlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Vocabulaires d'attributs
# ---------------------------------------------------------------------------
GENDERS = ["Homme", "Femme"]

# Groupes FairFace -- on parle de "groupe démographique / tonalité de peau"
SKIN_GROUPS = [
    "White", "Black", "Latino_Hispanic", "East Asian",
    "Southeast Asian", "Indian", "Middle Eastern",
]

# FairFace annote l'âge par tranches -> on prend le milieu de tranche pour
# obtenir un signal continu. Tranches hors [18, 70] filtrées.
FAIRFACE_AGE_MIDPOINTS = {
    "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 24.5, "30-39": 34.5,
    "40-49": 44.5, "50-59": 54.5, "60-69": 64.5, "more than 70": 75,
}

# Bornes réelles des tranches FairFace. Le point médian ci-dessus écrase
# l'information d'intervalle : après le filtre 18-70 ans, seules cinq valeurs
# d'âge subsistent (24,5 / 34,5 / ... / 64,5) et la tranche 10-19 disparaît
# entièrement alors qu'elle couvre 18 et 19 ans. Ces bornes permettent le mode
# `age_jitter` : on tire l'âge uniformément dans l'intersection de la tranche
# et de la plage demandée, ce qui restaure un support continu sur 18-70.
FAIRFACE_AGE_BINS = {
    "0-2": (0, 2), "3-9": (3, 9), "10-19": (10, 19), "20-29": (20, 29),
    "30-39": (30, 39), "40-49": (40, 49), "50-59": (50, 59),
    "60-69": (60, 69), "more than 70": (70, 90),
}

# Correspondance UTKFace (race: 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others)
UTK_TO_SKIN = {0: 0, 1: 1, 2: 3, 3: 5, 4: 2}


def normalize_age(age, min_age=18, max_age=70):
    """Âge en années -> scalaire [0, 1]."""
    return float(np.clip((age - min_age) / (max_age - min_age), 0.0, 1.0))


def denormalize_age(a, min_age=18, max_age=70):
    return a * (max_age - min_age) + min_age


def build_transforms(image_size, train=True):
    """Préprocessing : resize, crop, flip (augmentation), normalisation [-1, 1].

    La normalisation en [-1, 1] correspond au domaine de sortie tanh du
    générateur GAN et au bruit gaussien centré du DDPM.
    """
    tf = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if train:
        tf.append(transforms.RandomHorizontalFlip())
    tf += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
    ]
    return transforms.Compose(tf)


class FairFaceDataset(Dataset):
    """FairFace filtré sur la plage d'âge du sujet (18-70 ans)."""

    def __init__(self, root, split="train", image_size=64,
                 min_age=18, max_age=70, train_tf=True, age_jitter=False):
        self.root = pathlib.Path(root)
        csv = self.root / f"fairface_label_{split}.csv"
        df = pd.read_csv(csv)

        self.age_jitter = age_jitter
        if age_jitter:
            # On garde toute tranche qui INTERSECTE [min_age, max_age] et on
            # borne l'intervalle ; l'âge exact est tiré au vol dans __getitem__.
            lo = df["age"].map(lambda a: FAIRFACE_AGE_BINS[a][0])
            hi = df["age"].map(lambda a: FAIRFACE_AGE_BINS[a][1])
            df = df[(hi > min_age) & (lo < max_age)].copy()
            df["age_lo"] = lo[df.index].clip(lower=min_age)
            df["age_hi"] = hi[df.index].clip(upper=max_age)
            df["age_years"] = (df["age_lo"] + df["age_hi"]) / 2.0
        else:
            # Comportement historique (run principal) : point médian de tranche.
            df["age_years"] = df["age"].map(FAIRFACE_AGE_MIDPOINTS)
            df = df[(df["age_years"] >= min_age) & (df["age_years"] <= max_age)]
        df = df[df["race"].isin(SKIN_GROUPS)]
        self.df = df.reset_index(drop=True)

        self.min_age, self.max_age = min_age, max_age
        self.tf = build_transforms(image_size, train=train_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["file"]).convert("RGB")
        if self.age_jitter:
            years = np.random.uniform(row["age_lo"], row["age_hi"] + 1.0)
            years = min(years, self.max_age)
        else:
            years = row["age_years"]
        age = normalize_age(years, self.min_age, self.max_age)
        gender = 0 if row["gender"] == "Male" else 1
        skin = SKIN_GROUPS.index(row["race"])
        return self.tf(img), {
            "age": torch.tensor(age, dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
            "skin": torch.tensor(skin, dtype=torch.long),
        }


class UTKFaceDataset(Dataset):
    """UTKFace : âge exact dans le nom de fichier -> signal continu idéal."""

    def __init__(self, root, image_size=64, min_age=18, max_age=70,
                 train_tf=True):
        self.files = []
        for f in pathlib.Path(root).glob("**/*.jpg"):
            parts = f.name.split("_")
            if len(parts) < 4:
                continue  # fichiers mal nommés (connus dans UTKFace)
            try:
                age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if min_age <= age <= max_age:
                self.files.append((f, age, gender, race))
        self.min_age, self.max_age = min_age, max_age
        self.tf = build_transforms(image_size, train=train_tf)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f, age, gender, race = self.files[i]
        img = Image.open(f).convert("RGB")
        return self.tf(img), {
            "age": torch.tensor(normalize_age(age, self.min_age, self.max_age),
                                dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
            "skin": torch.tensor(UTK_TO_SKIN.get(race, 2), dtype=torch.long),
        }


def make_dataset(cfg, split="train", train_tf=True):
    if cfg.dataset == "fairface":
        return FairFaceDataset(cfg.root, split=split, image_size=cfg.image_size,
                               min_age=cfg.min_age, max_age=cfg.max_age,
                               train_tf=train_tf,
                               age_jitter=getattr(cfg, "age_jitter", False))
    if cfg.dataset == "utkface":
        return UTKFaceDataset(cfg.root, image_size=cfg.image_size,
                              min_age=cfg.min_age, max_age=cfg.max_age,
                              train_tf=train_tf)
    raise ValueError(f"Dataset inconnu : {cfg.dataset}")
