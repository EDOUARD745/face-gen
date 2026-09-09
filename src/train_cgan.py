"""Entraînement du cGAN baseline.

Usage :
    python -m src.train_cgan --data-root data/fairface --image-size 64
"""
import argparse
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from .config import Config
from .utils import get_device
from .data import make_dataset
from .models.cgan import Generator, Discriminator, d_loss_hinge, g_loss_hinge
from .train_ddpm import control_grid_attrs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/fairface")
    p.add_argument("--dataset", default="fairface", choices=["fairface", "utkface"])
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--out-dir", default="runs/cgan")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    cfg = Config()
    cfg.data.root, cfg.data.dataset = args.data_root, args.dataset
    cfg.data.image_size = args.image_size

    torch.manual_seed(cfg.train.seed)
    device = get_device()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tb")

    ds = make_dataset(cfg.data, split="train")
    print(f"Dataset : {len(ds)} images")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=cfg.data.num_workers, pin_memory=True,
                    drop_last=True)

    G = Generator(cfg.gan.latent_dim, image_size=args.image_size,
                  base=cfg.gan.base_channels).to(device)
    D = Discriminator(image_size=args.image_size,
                      base=cfg.gan.base_channels).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.gan.lr_g,
                             betas=(cfg.gan.beta1, cfg.gan.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.gan.lr_d,
                             betas=(cfg.gan.beta1, cfg.gan.beta2))

    step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        step, start_epoch = ck["step"], ck["epoch"]

    grid_attrs = control_grid_attrs(device)
    z_fixed = torch.randn(42, cfg.gan.latent_dim, device=device)

    for epoch in range(start_epoch, args.epochs):
        for imgs, attrs in dl:
            imgs = imgs.to(device, non_blocking=True)
            attrs = {k: v.to(device, non_blocking=True) for k, v in attrs.items()}
            b = imgs.shape[0]

            # ---- Discriminateur ----
            z = torch.randn(b, cfg.gan.latent_dim, device=device)
            with torch.no_grad():
                fake = G(z, attrs)
            loss_d = d_loss_hinge(D(imgs, attrs), D(fake, attrs))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- Générateur ----
            z = torch.randn(b, cfg.gan.latent_dim, device=device)
            fake = G(z, attrs)
            loss_g = g_loss_hinge(D(fake, attrs))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            step += 1

            if step % 100 == 0:
                writer.add_scalar("loss_d", loss_d.item(), step)
                writer.add_scalar("loss_g", loss_g.item(), step)
                print(f"epoch {epoch} step {step} "
                      f"D {loss_d.item():.3f} G {loss_g.item():.3f}")

            if step % 1000 == 0:
                G.eval()
                with torch.no_grad():
                    x = G(z_fixed, grid_attrs)
                G.train()
                grid = make_grid((x + 1) / 2, nrow=6)
                save_image(grid, out / f"grid_{step:07d}.png")
                writer.add_image("samples", grid, step)

            if step % 5000 == 0:
                torch.save({"G": G.state_dict(), "D": D.state_dict(),
                            "opt_g": opt_g.state_dict(),
                            "opt_d": opt_d.state_dict(),
                            "step": step, "epoch": epoch},
                           out / "ckpt_last.pt")

        torch.save({"G": G.state_dict(), "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "step": step, "epoch": epoch + 1}, out / "ckpt_last.pt")
    print("Entraînement terminé.")


if __name__ == "__main__":
    main()
