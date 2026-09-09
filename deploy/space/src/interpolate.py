"""Interpolation d'attributs (fonctionnalité exigée par le sujet).

Principe : avec DDIM déterministe (eta=0), le bruit initial x_T fixe
l'"identité" du visage. En gardant x_T constant et en faisant varier
continûment le vecteur d'attributs, on obtient le MÊME visage qui
vieillit / change de genre / de teinte de peau progressivement.

L'interpolation se fait dans l'espace d'embedding des attributs :
    c(alpha) = (1 - alpha) * c_A + alpha * c_B

Usage :
    python -m src.interpolate --ckpt runs/ddpm/ckpt_last.pt \
        --age-a 20 --age-b 68 --gender-a 0 --gender-b 0 \
        --skin-a 1 --skin-b 1 --frames 8 --out interp.png
"""
import argparse

import torch
from torchvision.utils import make_grid, save_image

from .utils import get_device, load_ddpm_from_ckpt
from .data import normalize_age


@torch.no_grad()
def interpolate(diffusion, attrs_a, attrs_b, frames=8, image_size=64,
                steps=50, guidance_scale=3.0, seed=0):
    """Retourne (frames, 3, H, W) : transition continue A -> B."""
    device = diffusion.device
    model = diffusion.model
    g = torch.Generator(device=device).manual_seed(seed)
    x_T = torch.randn(1, 3, image_size, image_size, device=device,
                      generator=g).repeat(frames, 1, 1, 1)

    emb_a = model.attr_embedder({k: v.to(device) for k, v in attrs_a.items()})
    emb_b = model.attr_embedder({k: v.to(device) for k, v in attrs_b.items()})
    alphas = torch.linspace(0, 1, frames, device=device)[:, None]
    cond = (1 - alphas) * emb_a + alphas * emb_b       # (frames, emb_dim)

    # Échantillonnage DDIM avec embeddings pré-calculés
    x = x_T
    times = torch.linspace(diffusion.timesteps - 1, 0, steps).long().tolist()
    pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]
    uncond = model.attr_embedder.null_condition(frames, device)
    for t_cur, t_next in pairs:
        t = torch.full((frames,), t_cur, device=device, dtype=torch.long)
        if guidance_scale == 1.0:
            eps = model(x, t, cond_emb=cond)
        else:
            eps = model(torch.cat([x, x]), torch.cat([t, t]),
                        cond_emb=torch.cat([cond, uncond]))
            eps_c, eps_u = eps.chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        ab_cur = diffusion.alpha_bar[t_cur]
        ab_next = diffusion.alpha_bar[t_next] if t_next >= 0 \
            else torch.tensor(1.0, device=device)
        x0 = ((x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()).clamp(-1, 1)
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x.clamp(-1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--age-a", type=float, default=20)
    p.add_argument("--age-b", type=float, default=68)
    p.add_argument("--gender-a", type=int, default=0)
    p.add_argument("--gender-b", type=int, default=0)
    p.add_argument("--skin-a", type=int, default=0)
    p.add_argument("--skin-b", type=int, default=0)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--out", default="interpolation.png")
    args = p.parse_args()

    device = get_device()
    _, diffusion, image_size = load_ddpm_from_ckpt(args.ckpt, device)

    def attrs(age, gender, skin):
        return {"age": torch.tensor([normalize_age(age)]),
                "gender": torch.tensor([gender]),
                "skin": torch.tensor([skin])}

    x = interpolate(diffusion,
                    attrs(args.age_a, args.gender_a, args.skin_a),
                    attrs(args.age_b, args.gender_b, args.skin_b),
                    frames=args.frames, image_size=image_size,
                    guidance_scale=args.guidance_scale, seed=args.seed)
    save_image(make_grid((x + 1) / 2, nrow=args.frames), args.out)
    print(f"Interpolation sauvegardée -> {args.out}")


if __name__ == "__main__":
    main()
