"""Baseline : GAN conditionnel (cGAN) type DCGAN + discriminateur à projection.

Sert de méthode de référence pour la comparaison exigée par le sujet.

  * Générateur : DCGAN (Radford et al., 2016) conditionné en concaténant
    l'embedding d'attributs au vecteur latent z (Mirza & Osindero, 2014).
  * Discriminateur : "projection discriminator" (Miyato & Koyama, 2018) --
    plus stable que la simple concaténation, la condition entre via un
    produit scalaire avec les features.
  * Le vecteur d'attributs réutilise le même encodage que le DDPM
    (âge continu + genre + peau), garantissant une comparaison équitable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import GENDERS, SKIN_GROUPS


class AttrEncoder(nn.Module):
    """(âge, genre, peau) -> vecteur dense (partagé G et D)."""

    def __init__(self, dim=128):
        super().__init__()
        self.gender_emb = nn.Embedding(len(GENDERS), dim)
        self.skin_emb = nn.Embedding(len(SKIN_GROUPS), dim)
        self.age_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(),
                                     nn.Linear(dim, dim))

    def forward(self, attrs):
        return (self.age_mlp(attrs["age"][:, None])
                + self.gender_emb(attrs["gender"])
                + self.skin_emb(attrs["skin"]))


class Generator(nn.Module):
    def __init__(self, latent_dim=128, attr_dim=128, base=64, image_size=64):
        super().__init__()
        self.attr_enc = AttrEncoder(attr_dim)
        # (taille amorce, nb up-samplings) ; image finale = amorce x 2**n_up.
        # Le DCGAN ne double la résolution que par 2 -> 48 = 3x2**4 (amorce 3x3).
        seed, n_up = {48: (3, 4), 64: (4, 4), 128: (4, 5)}[image_size]
        ch = base * 2 ** (n_up - 1)
        layers = [
            nn.ConvTranspose2d(latent_dim + attr_dim, ch, seed, 1, 0, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True),
        ]
        for _ in range(n_up - 1):
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch // 2), nn.ReLU(True),
            ]
            ch //= 2
        layers += [nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z, attrs):
        c = self.attr_enc(attrs)
        h = torch.cat([z, c], dim=1)[:, :, None, None]
        return self.net(h)


class Discriminator(nn.Module):
    """DCGAN + projection de la condition (Miyato & Koyama, 2018)."""

    def __init__(self, attr_dim=128, base=64, image_size=64):
        super().__init__()
        self.attr_enc = AttrEncoder(attr_dim)
        n_down = {48: 4, 64: 4, 128: 5}[image_size]
        layers = [nn.Conv2d(3, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = base
        for _ in range(n_down - 1):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch * 2), nn.LeakyReLU(0.2, True),
            ]
            ch *= 2
        self.features = nn.Sequential(*layers)          # -> (B, ch, 4, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(ch, 1)
        self.proj = nn.Linear(attr_dim, ch, bias=False)

    def forward(self, x, attrs):
        h = self.pool(self.features(x)).flatten(1)      # (B, ch)
        c = self.proj(self.attr_enc(attrs))             # (B, ch)
        # score = sortie inconditionnelle + produit scalaire condition/features
        return self.linear(h).squeeze(1) + (h * c).sum(dim=1)


def d_loss_hinge(d_real, d_fake):
    """Hinge loss (Lim & Ye, 2017) : plus stable que BCE pour les GANs."""
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()


def g_loss_hinge(d_fake):
    return -d_fake.mean()
