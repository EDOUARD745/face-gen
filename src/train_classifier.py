"""Classifieur d'attributs (ResNet-18) -- utilisé pour ÉVALUER la fidélité.

Trois têtes : âge (régression), genre (2 classes), peau (7 classes).
On génère N visages avec des attributs demandés, le classifieur prédit les
attributs perçus, et on mesure l'écart -> métrique de "fidélité aux
attributs" exigée par le sujet.

Usage :
    python -m src.train_classifier --data-root data/fairface --epochs 10
"""
import argparse
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from .config import Config
from .utils import get_device
from .data import make_dataset, GENDERS, SKIN_GROUPS


class AttributeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights="IMAGENET1K_V1")
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(feat, 1)                  # régression [0,1]
        self.gender_head = nn.Linear(feat, len(GENDERS))
        self.skin_head = nn.Linear(feat, len(SKIN_GROUPS))

    def forward(self, x):
        h = self.backbone(x)
        return (self.age_head(h).squeeze(1).sigmoid(),
                self.gender_head(h), self.skin_head(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="runs/classifier/attr_clf.pt")
    args = p.parse_args()

    device = get_device()
    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size

    train_ds = make_dataset(cfg.data, split="train")
    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)
    try:
        val_ds = make_dataset(cfg.data, split="val", train_tf=False)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    except Exception:
        val_dl = None

    model = AttributeClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        for imgs, attrs in dl:
            imgs = imgs.to(device)
            attrs = {k: v.to(device) for k, v in attrs.items()}
            age_p, gender_p, skin_p = model(imgs)
            loss = (mse(age_p, attrs["age"])
                    + ce(gender_p, attrs["gender"])
                    + ce(skin_p, attrs["skin"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")

        if val_dl is not None:
            model.eval()
            correct_g = correct_s = total = 0
            age_err = 0.0
            with torch.no_grad():
                for imgs, attrs in val_dl:
                    imgs = imgs.to(device)
                    attrs = {k: v.to(device) for k, v in attrs.items()}
                    age_p, gender_p, skin_p = model(imgs)
                    correct_g += (gender_p.argmax(1) == attrs["gender"]).sum().item()
                    correct_s += (skin_p.argmax(1) == attrs["skin"]).sum().item()
                    age_err += (age_p - attrs["age"]).abs().sum().item()
                    total += imgs.shape[0]
            print(f"  val: genre {100*correct_g/total:.1f}% | "
                  f"peau {100*correct_s/total:.1f}% | "
                  f"MAE âge {52*age_err/total:.1f} ans")  # 52 = plage 18-70

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Classifieur sauvegardé -> {out}")


if __name__ == "__main__":
    main()
