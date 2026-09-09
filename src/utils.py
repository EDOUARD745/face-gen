"""Utilitaires partagés."""
import torch


def get_device():
    """Meilleur device disponible : CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def config_from_ckpt(ck):
    """Reconstruit la Config depuis un checkpoint (config embarquée).

    Retourne la Config par défaut si le checkpoint est ancien / sans config
    exploitable — garantit la compatibilité modèle/checkpoint quand on
    entraîne avec un préset (ex : --preset mac).
    """
    from .config import Config, ModelConfig, DiffusionConfig, DataConfig

    cfg = Config()
    raw = ck.get("config")
    if isinstance(raw, dict):
        def detuple(d):
            return {k: tuple(v) if isinstance(v, list) else v
                    for k, v in d.items()}
        cfg.model = ModelConfig(**detuple(raw["model"]))
        cfg.diffusion = DiffusionConfig(**detuple(raw["diffusion"]))
        cfg.data = DataConfig(**raw["data"])
    return cfg


def load_ddpm_from_ckpt(ckpt_path, device, image_size=None):
    """Charge UNet (poids EMA) + GaussianDiffusion depuis un checkpoint.

    L'architecture est reconstruite depuis la config embarquée dans le
    checkpoint, pas depuis les valeurs par défaut.
    Retourne (model, diffusion, image_size).
    """
    from .diffusion import GaussianDiffusion
    from .models.unet import ConditionalUNet

    ck = torch.load(ckpt_path, map_location=device)
    cfg = config_from_ckpt(ck)
    if image_size is None:
        image_size = cfg.data.image_size
    model = ConditionalUNet(
        image_size=image_size,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        attn_resolutions=cfg.model.attn_resolutions,
        emb_dim=cfg.model.emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    model.load_state_dict(ck["ema"])  # poids EMA = meilleurs échantillons
    model.eval()
    diffusion = GaussianDiffusion(model, timesteps=cfg.diffusion.timesteps,
                                  schedule=cfg.diffusion.schedule,
                                  device=device)
    return model, diffusion, image_size
