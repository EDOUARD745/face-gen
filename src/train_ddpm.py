"""Entraînement du DDPM conditionnel.

Usage :
    python -m src.train_ddpm --data-root data/fairface --image-size 64 \
        --batch-size 64 --epochs 100

Sur Mac Apple Silicon (MPS), utiliser le préset allégé (~20M params, 48px,
~1-2 jours au lieu de 4-8) :
    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train_ddpm \
        --data-root data/fairface --preset mac

Reprise après interruption :
    python -m src.train_ddpm --resume runs/ddpm/ckpt_last.pt
"""
import argparse
import pathlib
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from .config import Config
from .utils import get_device
from .data import make_dataset, SKIN_GROUPS
from .diffusion import GaussianDiffusion, EMA
from .models.unet import ConditionalUNet


def control_grid_attrs(device):
    """Grille d'attributs fixe pour suivre visuellement l'entraînement :
    7 teintes de peau x (2 genres x 3 âges) -> 42 visages."""
    ages, genders, skins = [], [], []
    for s in range(len(SKIN_GROUPS)):
        for g in (0, 1):
            for a in (0.0, 0.5, 1.0):  # 18, 44, 70 ans
                ages.append(a); genders.append(g); skins.append(s)
    return {
        "age": torch.tensor(ages, device=device),
        "gender": torch.tensor(genders, device=device),
        "skin": torch.tensor(skins, device=device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out-dir", default="runs/ddpm")
    p.add_argument("--resume", default=None)
    p.add_argument("--preset", default=None, choices=["mac"],
                   help="'mac' : modèle allégé pour Apple Silicon (MPS)")
    args = p.parse_args()

    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size
    cfg.train.batch_size, cfg.train.epochs = args.batch_size, args.epochs
    cfg.train.lr, cfg.train.out_dir = args.lr, args.out_dir

    if args.preset == "mac":
        # ~20M params, 48px : qualité correcte en ~1-2 jours sur M1/M2/M3
        cfg.data.image_size = 48
        cfg.model.base_channels = 64
        cfg.model.channel_mults = (1, 2, 4)      # 48 -> 24 -> 12
        cfg.model.attn_resolutions = (12,)
        cfg.model.emb_dim = 256
        cfg.train.batch_size = min(args.batch_size, 32)
        if args.epochs == 100:                   # défaut non modifié par l'user
            cfg.train.epochs = 40
        if args.lr == 2e-4:                      # lr plus prudent sur MPS fp32
            cfg.train.lr = 1e-4
        print("Préset mac : 48px, base_channels=64, batch",
              cfg.train.batch_size, ",", cfg.train.epochs, "epochs, lr",
              cfg.train.lr)

    torch.manual_seed(cfg.train.seed)
    device = get_device()
    out = pathlib.Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    writer = SummaryWriter(out / "tb")

    ds = make_dataset(cfg.data, split="train")
    print(f"Dataset : {len(ds)} images ({cfg.data.dataset}, "
          f"{cfg.data.image_size}px, âges {cfg.data.min_age}-{cfg.data.max_age})")
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                    num_workers=cfg.data.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=cfg.data.num_workers > 0)

    model = ConditionalUNet(
        image_size=cfg.data.image_size,
        base_channels=cfg.model.base_channels,
        channel_mults=cfg.model.channel_mults,
        num_res_blocks=cfg.model.num_res_blocks,
        attn_resolutions=cfg.model.attn_resolutions,
        emb_dim=cfg.model.emb_dim,
        dropout=cfg.model.dropout,
    ).to(device)
    n_params = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"UNet conditionnel : {n_params:.1f}M paramètres")

    diffusion = GaussianDiffusion(model, timesteps=cfg.diffusion.timesteps,
                                  schedule=cfg.diffusion.schedule, device=device)
    ema = EMA(model, decay=cfg.train.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler(enabled=cfg.train.amp and device == "cuda")

    step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        ema.shadow = ck["ema"]
        opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"]
        print(f"Reprise depuis {args.resume} (epoch {start_epoch}, step {step})")

    def save_ckpt(epoch_val):
        """Sauvegarde atomique avec rotation : ckpt_last -> ckpt_prev.
        Si un NaN corrompt les poids, ckpt_prev.pt reste sain."""
        tmp = out / "ckpt_tmp.pt"
        torch.save({"model": model.state_dict(), "ema": ema.shadow,
                    "opt": opt.state_dict(), "step": step,
                    "epoch": epoch_val, "config": asdict(cfg)}, tmp)
        last = out / "ckpt_last.pt"
        if last.exists():
            last.replace(out / "ckpt_prev.pt")
        tmp.replace(last)

    grid_attrs = control_grid_attrs(device)
    nan_streak = 0
    t0 = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        for imgs, attrs in dl:
            imgs = imgs.to(device, non_blocking=True)
            attrs = {k: v.to(device, non_blocking=True) for k, v in attrs.items()}

            with torch.amp.autocast("cuda", enabled=cfg.train.amp and device == "cuda"):
                loss = diffusion.loss(imgs, attrs,
                                      cond_drop_prob=cfg.diffusion.cond_drop_prob)

            # Garde-fou NaN n°1 : loss non finie -> batch ignoré, poids intacts
            if not torch.isfinite(loss):
                nan_streak += 1
                print(f"[garde-fou] loss non finie au step {step} "
                      f"({nan_streak} consécutifs) — batch ignoré")
                if nan_streak >= 20:
                    raise RuntimeError(
                        "20 loss non finies consécutives : entraînement "
                        "instable, reprendre depuis runs/ddpm/ckpt_prev.pt "
                        "avec --lr plus bas.")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.train.grad_clip)

            # Garde-fou NaN n°2 : gradients non finis -> pas d'update
            if not torch.isfinite(grad_norm):
                nan_streak += 1
                print(f"[garde-fou] gradients non finis au step {step} "
                      f"({nan_streak} consécutifs) — update ignoré")
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue

            nan_streak = 0
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1

            if step % cfg.train.log_every == 0:
                ips = cfg.train.log_every * cfg.train.batch_size / (time.time() - t0)
                t0 = time.time()
                writer.add_scalar("loss", loss.item(), step)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"({ips:.0f} img/s)")

            if step % cfg.train.sample_every == 0:
                model.eval()
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                x = diffusion.sample_ddim(
                    grid_attrs, (42, 3, cfg.data.image_size, cfg.data.image_size),
                    steps=cfg.diffusion.ddim_steps,
                    guidance_scale=cfg.diffusion.guidance_scale)
                grid = make_grid((x + 1) / 2, nrow=6)
                save_image(grid, out / f"grid_{step:07d}.png")
                writer.add_image("samples", grid, step)
                model.load_state_dict(backup)
                model.train()

            if step % cfg.train.ckpt_every == 0:
                save_ckpt(epoch)

        save_ckpt(epoch + 1)
    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
