"""Processus de diffusion gaussien (DDPM) + échantillonnage DDIM + CFG.

Références implémentées :
  * DDPM              -- Ho, Jain & Abbeel (2020)
  * Cosine schedule   -- Nichol & Dhariwal (2021), "Improved DDPM"
  * DDIM              -- Song, Meng & Ermon (2021) : échantillonnage rapide
                         déterministe en ~50 pas au lieu de 1000
  * Classifier-Free Guidance -- Ho & Salimans (2022) :
        eps_guidé = eps_uncond + w * (eps_cond - eps_uncond)
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm


def make_beta_schedule(schedule, timesteps):
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule == "cosine":
        # alpha_bar(t) = cos^2(((t/T)+s)/(1+s) * pi/2), s = 0.008
        s = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(max=0.999).float()
    raise ValueError(schedule)


class GaussianDiffusion:
    """Encapsule q(x_t|x_0), la perte, et p(x_{t-1}|x_t) DDPM/DDIM."""

    def __init__(self, model, timesteps=1000, schedule="cosine", device="cuda"):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        betas = make_beta_schedule(schedule, timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = alpha_bar.sqrt()
        self.sqrt_1mab = (1 - alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def q_sample(self, x0, t, noise):
        """Processus forward : x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps."""
        return (self.sqrt_ab[t][:, None, None, None] * x0
                + self.sqrt_1mab[t][:, None, None, None] * noise)

    def loss(self, x0, attrs, cond_drop_prob=0.1, independent_drop=False,
             joint_drop_prob=0.05):
        """Perte simple L = ||eps - eps_theta(x_t, t, c)||^2 (Ho et al.).

        `independent_drop` masque chaque attribut indépendamment plutôt qu'en
        bloc. Le réseau voit alors des exemples où l'âge est la seule condition
        disponible et ne peut plus l'ignorer en s'appuyant sur le genre et la
        peau. Un masquage conjoint résiduel (`joint_drop_prob`) préserve la
        branche entièrement inconditionnelle dont le CFG standard a besoin.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        if independent_drop:
            joint = torch.rand(b, device=x0.device) < joint_drop_prob
            drop_mask = {k: (torch.rand(b, device=x0.device) < cond_drop_prob) | joint
                         for k in ("age", "gender", "skin")}
        else:
            # masque CFG historique : on retire toute la condition sur ~10% du batch
            drop_mask = torch.rand(b, device=x0.device) < cond_drop_prob
        pred = self.model(x_t, t, attrs=attrs, drop_mask=drop_mask)
        return F.mse_loss(pred, noise)

    # ------------------------------------------------------------------
    # Échantillonnage
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _guided_eps(self, x, t, attrs, guidance_scale):
        """Prédiction de bruit avec classifier-free guidance.

        `guidance_scale` est soit un scalaire (CFG standard, deux branches),
        soit un dict {age, gender, skin} : guidage COMPOSITIONNEL, une échelle
        par attribut (Liu et al., 2022). Utile quand un attribut est moins bien
        suivi que les autres : on peut pousser son guidage sans sursaturer les
        autres, ce que l'échelle unique ne permet pas.
        """
        if isinstance(guidance_scale, dict):
            return self._composed_eps(x, t, attrs, guidance_scale)
        if guidance_scale == 1.0:
            return self.model(x, t, attrs=attrs)
        b = x.shape[0]
        cond = self.model.attr_embedder(attrs)
        uncond = self.model.attr_embedder.null_condition(b, x.device)
        eps = self.model(
            torch.cat([x, x]), torch.cat([t, t]),
            cond_emb=torch.cat([cond, uncond]),
        )
        eps_c, eps_u = eps.chunk(2)
        return eps_u + guidance_scale * (eps_c - eps_u)

    @torch.no_grad()
    def _composed_eps(self, x, t, attrs, scales):
        """eps = eps_null + somme_k w_k (eps_{k seul} - eps_null).

        Nécessite un modèle entraîné avec masquage indépendant : les branches
        « un seul attribut » seraient hors distribution autrement.
        """
        b = x.shape[0]
        E = self.model.attr_embedder
        keys = [k for k in ("age", "gender", "skin") if scales.get(k, 0.0)]
        conds = [E.condition_keeping(attrs, (k,)) for k in keys]
        uncond = E.null_condition(b, x.device)
        n = len(keys) + 1
        eps = self.model(
            x.repeat(n, 1, 1, 1), t.repeat(n),
            cond_emb=torch.cat(conds + [uncond]),
        )
        chunks = eps.chunk(n)
        eps_u = chunks[-1]
        out = eps_u
        for k, e in zip(keys, chunks[:-1]):
            out = out + scales[k] * (e - eps_u)
        return out

    @torch.no_grad()
    def sample_ddpm(self, attrs, shape, guidance_scale=3.0, progress=True):
        """Échantillonnage ancestral complet (T pas). Lent mais exact."""
        x = torch.randn(shape, device=self.device)
        steps = reversed(range(self.timesteps))
        if progress:
            steps = tqdm(list(steps), desc="DDPM sampling")
        for i in steps:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = self._guided_eps(x, t, attrs, guidance_scale)
            ab, a, b_ = self.alpha_bar[i], self.alphas[i], self.betas[i]
            mean = (x - b_ / (1 - ab).sqrt() * eps) / a.sqrt()
            if i > 0:
                x = mean + b_.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)

    @torch.no_grad()
    def sample_ddim(self, attrs, shape, steps=50, eta=0.0,
                    guidance_scale=3.0, progress=False, x_T=None,
                    progress_cb=None):
        """Échantillonnage DDIM : ~50 pas -> génération quasi temps réel.

        eta=0 -> déterministe : même seed + mêmes attributs = même visage,
        propriété exploitée pour l'interpolation d'attributs.
        """
        x = x_T if x_T is not None else torch.randn(shape, device=self.device)
        times = torch.linspace(self.timesteps - 1, 0, steps).long().tolist()
        pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]
        it = tqdm(pairs, desc="DDIM sampling") if progress else pairs
        done_steps = 0
        for t_cur, t_next in it:
            t = torch.full((shape[0],), t_cur, device=self.device, dtype=torch.long)
            eps = self._guided_eps(x, t, attrs, guidance_scale)
            ab_cur = self.alpha_bar[t_cur]
            ab_next = self.alpha_bar[t_next] if t_next >= 0 \
                else torch.tensor(1.0, device=self.device)
            x0_pred = ((x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()).clamp(-1, 1)
            sigma = eta * ((1 - ab_next) / (1 - ab_cur)).sqrt() \
                * (1 - ab_cur / ab_next).sqrt()
            dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_next.sqrt() * x0_pred + dir_xt
            if eta > 0 and t_next >= 0:
                x = x + sigma * torch.randn_like(x)
            done_steps += 1
            if progress_cb is not None:
                progress_cb(done_steps, len(pairs))
        return x.clamp(-1, 1)


class EMA:
    """Exponential Moving Average des poids -- stabilise les échantillons.

    Standard en diffusion : les poids EMA donnent un FID nettement meilleur
    que les poids bruts (Ho et al., 2020, annexe).
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
