"""UNet conditionnel pour DDPM.

Architecture inspirée de Ho et al. (2020) et Dhariwal & Nichol (2021),
adaptée au contrôle d'attributs démographiques :

  * temps t        -> embedding sinusoïdal + MLP
  * âge (continu)  -> embedding sinusoïdal du scalaire normalisé + MLP
  * genre (classe) -> nn.Embedding
  * peau  (classe) -> nn.Embedding

Les 4 embeddings sont sommés puis injectés dans chaque bloc résiduel
(modulation additive type "adaptive group norm" simplifiée).

Classifier-Free Guidance : chaque attribut possède un embedding "null"
appris. À l'entraînement, la condition complète est remplacée par les
embeddings null avec probabilité cond_drop_prob, ce qui permet au même
réseau d'estimer eps(x_t) ET eps(x_t | c).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import GENDERS, SKIN_GROUPS


def sinusoidal_embedding(x, dim):
    """Embedding sinusoïdal (Transformer / DDPM). x : (B,) float."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=x.device).float() / half
    )
    args = x[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AttributeEmbedder(nn.Module):
    """Encode (âge, genre, peau) -> vecteur de conditionnement.

    Gère le masquage pour le classifier-free guidance via des embeddings
    "null" appris (un par attribut).
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.age_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.gender_emb = nn.Embedding(len(GENDERS), emb_dim)
        self.skin_emb = nn.Embedding(len(SKIN_GROUPS), emb_dim)
        # Embeddings "condition absente" (CFG)
        self.null_age = nn.Parameter(torch.zeros(emb_dim))
        self.null_gender = nn.Parameter(torch.zeros(emb_dim))
        self.null_skin = nn.Parameter(torch.zeros(emb_dim))

    def parts(self, attrs):
        """Embedding de chaque attribut, séparément."""
        # âge continu x 1000 pour couvrir la gamme de fréquences sinusoïdales
        return {
            "age": self.age_mlp(
                sinusoidal_embedding(attrs["age"] * 1000.0, self.emb_dim)),
            "gender": self.gender_emb(attrs["gender"]),
            "skin": self.skin_emb(attrs["skin"]),
        }

    def forward(self, attrs, drop_mask=None):
        """attrs : dict(age (B,), gender (B,), skin (B,)).

        drop_mask : (B,) bool -- True = TOUTE la condition est masquée, ou
        dict {attribut: (B,) bool} -- masquage indépendant par attribut. Le
        masquage indépendant est ce qui force le réseau à exploiter chaque
        attribut isolément : sommés puis masqués en bloc, deux attributs forts
        (genre, peau) suffisent à annuler la perte et un attribut faible
        (l'âge) peut être ignoré sans jamais coûter.
        """
        parts = self.parts(attrs)
        nulls = {"age": self.null_age, "gender": self.null_gender,
                 "skin": self.null_skin}

        if drop_mask is not None:
            masks = drop_mask if isinstance(drop_mask, dict) else \
                {k: drop_mask for k in parts}
            for k, v in parts.items():
                m = masks[k][:, None].float()
                parts[k] = m * nulls[k][None] + (1 - m) * v
        return parts["age"] + parts["gender"] + parts["skin"]

    def condition_keeping(self, attrs, keep):
        """Condition où seuls les attributs de `keep` sont renseignés.

        Utilisé par le guidage compositionnel : la branche « âge seul » mesure
        la direction propre à l'âge, indépendamment du genre et de la peau.
        """
        b = attrs["age"].shape[0]
        dev = attrs["age"].device
        masks = {k: torch.full((b,), k not in keep, dtype=torch.bool, device=dev)
                 for k in ("age", "gender", "skin")}
        return self.forward(attrs, masks)

    def null_condition(self, batch_size, device):
        """Vecteur de conditionnement 'vide' pour la branche inconditionnelle."""
        return (self.null_age + self.null_gender + self.null_skin)[None].expand(
            batch_size, -1
        ).to(device)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                     else nn.Identity())

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    """UNet epsilon-prédicteur conditionné (temps + attributs)."""

    def __init__(self, image_size=64, in_ch=3, base_channels=128,
                 channel_mults=(1, 2, 2, 4), num_res_blocks=2,
                 attn_resolutions=(16, 8), emb_dim=512, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        time_dim = base_channels

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.time_dim = time_dim
        self.attr_embedder = AttributeEmbedder(emb_dim)

        self.stem = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        # ----- Encodeur -----
        self.down_blocks = nn.ModuleList()
        chans = [base_channels]
        ch = base_channels
        res = image_size
        for i, mult in enumerate(channel_mults):
            out = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([
                    ResBlock(ch, out, emb_dim, dropout),
                    SelfAttention(out) if res in attn_resolutions else nn.Identity(),
                ])
                self.down_blocks.append(block)
                ch = out
                chans.append(ch)
            if i < len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch), nn.Identity()]))
                chans.append(ch)
                res //= 2

        # ----- Goulot -----
        self.mid1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid2 = ResBlock(ch, ch, emb_dim, dropout)

        # ----- Décodeur -----
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out = base_channels * mult
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([
                    ResBlock(ch + chans.pop(), out, emb_dim, dropout),
                    SelfAttention(out) if res in attn_resolutions else nn.Identity(),
                ])
                self.up_blocks.append(block)
                ch = out
            if i > 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch), nn.Identity()]))
                res *= 2

        self.head = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x, t, attrs=None, drop_mask=None, cond_emb=None):
        """x (B,3,H,W), t (B,), attrs dict ou cond_emb pré-calculé."""
        emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        if cond_emb is None:
            cond_emb = self.attr_embedder(attrs, drop_mask)
        emb = emb + cond_emb

        h = self.stem(x)
        skips = [h]
        for block in self.down_blocks:
            first, second = block
            if isinstance(first, Downsample):
                h = first(h)
            else:
                h = second(first(h, emb)) if not isinstance(second, nn.Identity) \
                    else first(h, emb)
            skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)

        for block in self.up_blocks:
            first, second = block
            if isinstance(first, Upsample):
                h = first(h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = second(first(h, emb)) if not isinstance(second, nn.Identity) \
                    else first(h, emb)

        return self.head(h)
