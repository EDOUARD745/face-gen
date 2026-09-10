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
    p.add_argument("--preset", default=None, choices=["mac", "mac64", "mac96"],
                   help="'mac' : 48px (modèle v1/v2) ; 'mac64' / 'mac96' : "
                        "même architecture à plus haute résolution. "
                        "L'attention reste au troisième étage (résolution/4), "
                        "ce qui garde la structure des poids identique et "
                        "permet de reprendre un checkpoint 48px.")
    p.add_argument("--independent-drop", action="store_true",
                   help="masque chaque attribut indépendamment (CFG) : force "
                        "le réseau à exploiter l'âge seul")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="décroissance EMA (défaut 0.9999 ; 0.999 pour un "
                        "fine-tune court, sinon l'EMA masque le progrès)")
    p.add_argument("--age-jitter", action="store_true",
                   help="tire l'âge dans la tranche FairFace annotée au lieu "
                        "du point médian : restaure un support d'âge continu "
                        "sur 18-70 (cf. src/data.py)")
    args = p.parse_args()

    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size
    cfg.train.batch_size, cfg.train.epochs = args.batch_size, args.epochs
    cfg.train.lr, cfg.train.out_dir = args.lr, args.out_dir
    cfg.data.age_jitter = args.age_jitter

    if args.resume:
        # L'architecture doit correspondre au checkpoint : on la relit depuis la
        # config embarquée (sinon une reprise sans --preset reconstruit un UNet
        # 64px/128 canaux et le load_state_dict échoue).
        from .utils import config_from_ckpt
        _ck = torch.load(args.resume, map_location="cpu")
        _prev = config_from_ckpt(_ck)
        cfg.model = _prev.model
        cfg.diffusion = _prev.diffusion
        cfg.data.image_size = _prev.data.image_size
        print(f"Config d'architecture reprise du checkpoint : "
              f"{cfg.data.image_size}px, base_channels={cfg.model.base_channels}")
        del _ck

    if args.preset:
        # Même squelette à trois étages pour les trois présets. L'attention est
        # placée à résolution/4, donc toujours au troisième étage : la liste des
        # modules — et donc les clés du state_dict — reste identique d'une
        # résolution à l'autre, ce qui autorise la reprise d'un checkpoint 48px
        # pour un entraînement en 64 ou 96 px (redimensionnement progressif).
        size = {"mac": 48, "mac64": 64, "mac96": 96}[args.preset]
        default_batch = {"mac": 32, "mac64": 32, "mac96": 16}[args.preset]
        cfg.data.image_size = size
        cfg.model.base_channels = 64
        cfg.model.channel_mults = (1, 2, 4)      # size -> size/2 -> size/4
        cfg.model.attn_resolutions = (size // 4,)
        cfg.model.emb_dim = 256
        cfg.train.batch_size = min(args.batch_size, default_batch)
        if args.epochs == 100:                   # défaut non modifié par l'user
            cfg.train.epochs = 40
        if args.lr == 2e-4:                      # lr plus prudent sur MPS fp32
            cfg.train.lr = 1e-4
        print(f"Préset {args.preset} : {size}px, base_channels=64, attention à "
              f"{size // 4}px, batch {cfg.train.batch_size}, "
              f"{cfg.train.epochs} epochs, lr {cfg.train.lr}")

    torch.manual_seed(cfg.train.seed)
    device = get_device()
    # Après --resume et --preset, qui peuvent remplacer cfg.diffusion en bloc.
    if args.independent_drop:
        cfg.diffusion.independent_cond_drop = True
    if args.ema_decay is not None:
        cfg.train.ema_decay = args.ema_decay
    print(f"Masquage indépendant par attribut : "
          f"{cfg.diffusion.independent_cond_drop} | EMA {cfg.train.ema_decay}")

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
                loss = diffusion.loss(
                    imgs, attrs,
                    cond_drop_prob=cfg.diffusion.cond_drop_prob,
                    independent_drop=cfg.diffusion.independent_cond_drop,
                    joint_drop_prob=cfg.diffusion.joint_drop_prob)

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
