"""Configuration centralisée du projet.

Toutes les expériences sont pilotées par ce fichier afin de garantir
la reproductibilité (critère d'évaluation du projet).
"""
from dataclasses import dataclass, field, asdict
import json
import pathlib


@dataclass
class DataConfig:
    dataset: str = "fairface"          # "fairface" ou "utkface"
    root: str = "data/fairface"        # dossier contenant images + csv de labels
    image_size: int = 64               # 64 pour itérer vite, 128 pour le rendu final
    min_age: int = 18                  # bornes exigées par le sujet
    max_age: int = 70
    num_workers: int = 4
    # Restaure un support d'âge continu (tirage dans la tranche annotée) au
    # lieu des cinq points médians. Défaut False = configuration du run
    # principal, conservée pour la reproductibilité.
    age_jitter: bool = False


@dataclass
class DiffusionConfig:
    timesteps: int = 1000              # T du DDPM (Ho et al., 2020)
    schedule: str = "cosine"           # "linear" ou "cosine" (Nichol & Dhariwal, 2021)
    # Classifier-Free Guidance (Ho & Salimans, 2022)
    cond_drop_prob: float = 0.1        # proba de masquer la condition à l'entraînement
    guidance_scale: float = 3.0        # w à l'échantillonnage (1.0 = pas de guidage)
    ddim_steps: int = 50               # échantillonnage rapide quasi temps réel
    ddim_eta: float = 0.0


@dataclass
class ModelConfig:
    base_channels: int = 128
    channel_mults: tuple = (1, 2, 2, 4)   # 64 -> 32 -> 16 -> 8
    attn_resolutions: tuple = (16, 8)     # self-attention à ces résolutions
    num_res_blocks: int = 2
    dropout: float = 0.1
    emb_dim: int = 512                    # dimension embedding temps + attributs


@dataclass
class GANConfig:
    latent_dim: int = 128
    base_channels: int = 64
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 2e-4
    epochs: int = 100
    ema_decay: float = 0.9999          # EMA des poids : indispensable en diffusion
    grad_clip: float = 1.0
    amp: bool = True                   # mixed precision -> ~2x plus rapide sur GPU
    log_every: int = 100
    sample_every: int = 2000           # génère une grille d'images de contrôle
    ckpt_every: int = 5000
    out_dir: str = "runs/ddpm"
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path):
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str))

    @staticmethod
    def load(path):
        raw = json.loads(pathlib.Path(path).read_text())
        return Config(
            data=DataConfig(**raw["data"]),
            diffusion=DiffusionConfig(**{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in raw["diffusion"].items()}),
            model=ModelConfig(**{k: tuple(v) if isinstance(v, list) else v
                                 for k, v in raw["model"].items()}),
            gan=GANConfig(**raw["gan"]),
            train=TrainConfig(**raw["train"]),
        )
