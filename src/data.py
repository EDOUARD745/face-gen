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
                 min_age=18, max_age=70, train_tf=True):
        self.root = pathlib.Path(root)
        csv = self.root / f"fairface_label_{split}.csv"
        df = pd.read_csv(csv)

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
        age = normalize_age(row["age_years"], self.min_age, self.max_age)
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
                               train_tf=train_tf)
    if cfg.dataset == "utkface":
        return UTKFaceDataset(cfg.root, image_size=cfg.image_size,
                              min_age=cfg.min_age, max_age=cfg.max_age,
                              train_tf=train_tf)
    raise ValueError(f"Dataset inconnu : {cfg.dataset}")
