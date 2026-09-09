"""VISAGE - Interface de génération de visages avec contrôle d'attributs.

Lancement :
    python app.py --ckpt runs/ddpm/ckpt_last.pt [--cgan-ckpt runs/cgan/ckpt_last.pt]

Sans checkpoint (démo UI seule) :
    python app.py --demo
"""
import argparse
import pathlib

import numpy as np
import torch
import gradio as gr

from src.config import Config
from src.utils import get_device, load_ddpm_from_ckpt
from src.data import normalize_age, SKIN_GROUPS
from src.sampling import (load_calibration, resample_artifacts,
                          sample_best_of)
from src.models.cgan import Generator
from src.interpolate import interpolate as interp_fn

# ---------------------------------------------------------------------------
# Style - thème sombre "aurora glass", édition premium
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #06070d;
  --panel: rgba(255,255,255,0.032);
  --border: rgba(255,255,255,0.08);
  --accent: #7c6cff;
  --accent2: #00e5c3;
  --accent3: #ff5e94;
  --text-dim: rgba(255,255,255,.45);
}

/* ---------- Fond : aurores animées + grain ---------- */
html, body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
}
.gradio-container::before {
  content: ''; position: fixed; inset: -20%; z-index: 0;
  pointer-events: none;
  background:
    radial-gradient(ellipse 42% 30% at 18% 8%,  rgba(124,108,255,.20), transparent 65%),
    radial-gradient(ellipse 38% 26% at 82% 12%, rgba(0,229,195,.13),  transparent 65%),
    radial-gradient(ellipse 36% 24% at 50% 95%, rgba(255,94,148,.09), transparent 65%);
  animation: aurora 26s ease-in-out infinite alternate;
}
@keyframes aurora {
  0%   { transform: translate(0,0) scale(1); }
  50%  { transform: translate(2.5%,-1.5%) scale(1.06); }
  100% { transform: translate(-2%,1.5%) scale(1.03); }
}
.gradio-container::after {
  content: ''; position: fixed; inset: 0; z-index: 0;
  pointer-events: none; opacity: .05;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

/* ---------- Hero ---------- */
#hero { text-align: center; padding: 34px 0 10px; position: relative; }
#hero h1 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 3.4rem; font-weight: 700; letter-spacing: .3em; margin: 0;
  background: linear-gradient(100deg,#fff 10%,#a89bff 35%,#00e5c3 60%,#ff9ebc 85%,#fff 110%);
  background-size: 220% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 9s linear infinite;
  text-shadow: 0 0 44px rgba(124,108,255,.25);
}
@keyframes shimmer { to { background-position: 220% center; } }
#hero .sub {
  color: var(--text-dim); letter-spacing: .14em; font-size: .78rem;
  text-transform: uppercase; margin-top: 10px;
}
#hero .badges { margin-top: 14px; display: flex; gap: 8px;
  justify-content: center; flex-wrap: wrap; }
#hero .badge {
  font-size: .68rem; letter-spacing: .1em; text-transform: uppercase;
  color: rgba(255,255,255,.72); padding: 4px 12px; border-radius: 999px;
  border: 1px solid var(--border);
  background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
  backdrop-filter: blur(8px);
}
#hero .badge b { color: var(--accent2); font-weight: 600; }

/* ---------- Panneaux verre ---------- */
/* NB : pas de backdrop-filter ici -- il crée un contexte d'empilement qui
   fait passer les menus déroulants DERRIÈRE les panneaux voisins. */
.gr-group, .gr-panel, .block {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  overflow: visible !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.055),
              0 10px 34px rgba(0,0,0,.32) !important;
  transition: border-color .25s ease;
}
.gr-group:hover, .gr-panel:hover { border-color: rgba(255,255,255,.15) !important; }

/* ---------- Menus déroulants (Dropdown) ---------- */
/* Panneau qui contient un menu ouvert : passe au premier plan */
.block:has(ul.options), .form:has(ul.options), .wrap:has(ul.options) {
  position: relative; z-index: 500 !important;
}
ul.options, .options {
  z-index: 1000 !important;
  background: #12141f !important;
  border: 1px solid rgba(124,108,255,.4) !important;
  border-radius: 12px !important;
  box-shadow: 0 12px 40px rgba(0,0,0,.6) !important;
}
ul.options li, .options .item {
  color: rgba(255,255,255,.85) !important;
}
ul.options li:hover, .options .item:hover,
ul.options li.selected, .options .item.selected {
  background: rgba(124,108,255,.22) !important;
  color: #fff !important;
}

/* ---------- Boutons ---------- */
button.primary, .gr-button-primary {
  position: relative; overflow: hidden;
  background: linear-gradient(120deg,#7c6cff,#5e8bff 45%,#00c3a8) !important;
  background-size: 160% auto !important;
  border: none !important; border-radius: 13px !important;
  font-weight: 600 !important; letter-spacing: .05em;
  box-shadow: 0 4px 26px rgba(124,108,255,.4),
              inset 0 1px 0 rgba(255,255,255,.25) !important;
  transition: transform .16s ease, box-shadow .16s ease,
              background-position .4s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
  transform: translateY(-1.5px);
  background-position: 90% center !important;
  box-shadow: 0 8px 38px rgba(124,108,255,.6),
              inset 0 1px 0 rgba(255,255,255,.3) !important;
}
button.primary:active { transform: translateY(0) scale(.985); }
button.primary::after {
  content: ''; position: absolute; top: 0; left: -80%;
  width: 55%; height: 100%; transform: skewX(-22deg);
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.28), transparent);
  transition: left .5s ease;
}
button.primary:hover::after { left: 130%; }
button.secondary, .gr-button-secondary {
  border: 1px solid var(--border) !important;
  background: rgba(255,255,255,.045) !important;
  border-radius: 13px !important; backdrop-filter: blur(8px);
  transition: border-color .2s ease, background .2s ease !important;
}
button.secondary:hover { border-color: rgba(124,108,255,.55) !important;
  background: rgba(124,108,255,.12) !important; }

/* ---------- Contrôles ---------- */
input[type=range] { accent-color: var(--accent) !important; }
input[type=range]::-webkit-slider-thumb {
  box-shadow: 0 0 12px rgba(124,108,255,.8);
}
label span { color: rgba(255,255,255,.78) !important; font-weight: 500 !important; }
input:focus, select:focus, textarea:focus {
  outline: none !important;
  border-color: rgba(124,108,255,.6) !important;
  box-shadow: 0 0 0 3px rgba(124,108,255,.18) !important;
}
input[type=checkbox], input[type=radio] { accent-color: var(--accent) !important; }

/* ---------- Onglets pilules ---------- */
.tabs > div:first-child, .tab-nav {
  border-bottom: none !important; gap: 6px;
  padding: 6px; border-radius: 999px;
  background: rgba(255,255,255,.03);
  border: 1px solid var(--border);
  width: fit-content; margin: 0 auto 10px;
}
.tabs button, .tab-nav button {
  border-radius: 999px !important; border: none !important;
  color: var(--text-dim) !important; font-weight: 500 !important;
  letter-spacing: .02em; padding: 8px 18px !important;
  transition: color .2s ease, background .2s ease !important;
}
.tabs button:hover { color: rgba(255,255,255,.85) !important; }
.tabs button.selected, .tab-nav button.selected {
  color: #fff !important;
  background: linear-gradient(120deg, rgba(124,108,255,.32), rgba(0,195,168,.22)) !important;
  border: none !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.12),
              0 2px 14px rgba(124,108,255,.28) !important;
}

/* ---------- Galeries ---------- */
.gallery, .grid-wrap { border-radius: 16px !important; }
.gallery img, .grid-wrap img, .thumbnail-item img {
  border-radius: 10px !important;
  transition: transform .22s ease, box-shadow .22s ease !important;
}
.thumbnail-item { border-radius: 12px !important; overflow: hidden;
  border: 1px solid transparent !important;
  transition: border-color .2s ease !important; }
.thumbnail-item:hover { border-color: rgba(124,108,255,.55) !important; }
.thumbnail-item:hover img { transform: scale(1.045);
  box-shadow: 0 6px 22px rgba(0,0,0,.45); }

/* ---------- Divers ---------- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(124,108,255,.32); border-radius: 999px;
  border: 2px solid var(--bg);
}
::-webkit-scrollbar-thumb:hover { background: rgba(124,108,255,.55); }
.prose h3 { color: #cfc8ff !important; }
.prose code {
  background: rgba(124,108,255,.14) !important;
  border: 1px solid rgba(124,108,255,.22);
  border-radius: 6px; padding: 1px 6px;
}
footer { display: none !important; }

@media (prefers-reduced-motion: reduce) {
  .gradio-container::before, #hero h1 { animation: none !important; }
  * { transition: none !important; }
}
"""

SKIN_LABELS_FR = {
    "White": "Claire (Europe)", "Black": "Foncée (Afrique)",
    "Latino_Hispanic": "Métisse (Latino)", "East Asian": "Asie de l'Est",
    "Southeast Asian": "Asie du Sud-Est", "Indian": "Asie du Sud (Inde)",
    "Middle Eastern": "Moyen-Orient",
}
SKIN_CHOICES = [f"{SKIN_LABELS_FR[s]}" for s in SKIN_GROUPS]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class Engine:
    @staticmethod
    def _default_calibration(ckpt):
        """Calibration voisine du checkpoint chargé, pas celle d'un autre run.

        Chaque modèle a sa propre fonction de réponse en âge : appliquer la
        table d'un autre run corrigerait dans le vide.
        """
        if not ckpt:
            return None
        cand = pathlib.Path(ckpt).parent / "age_calibration.json"
        return str(cand) if cand.exists() else None

    def __init__(self, ckpt=None, cgan_ckpt=None, image_size=64, demo=False,
                 calibration="auto", best_of=1, clf=None,
                 reject_artifacts=True):
        if calibration == "auto":
            calibration = self._default_calibration(ckpt)
        self.demo = demo or ckpt is None
        self.image_size = image_size
        self.device = get_device()
        self.diffusion, self.cgan = None, None
        # Calibration post-hoc de la condition d'âge (cf. src/sampling.py) :
        # corrige la compression de la réponse en âge sans ré-entraînement.
        self.calibration = load_calibration(calibration)
        self.best_of = max(1, int(best_of))
        # Démonstrateur uniquement : ~2-3 % des tirages sortent de la plage
        # colorimétrique des images réelles (visages verts). On les régénère.
        self.reject_artifacts = reject_artifacts
        self.clf = None
        if self.best_of > 1 and clf:
            from src.train_classifier import AttributeClassifier
            self.clf = AttributeClassifier().to(self.device)
            self.clf.load_state_dict(torch.load(clf, map_location=self.device))
            self.clf.eval()
        if not self.demo:
            # Architecture reconstruite depuis la config embarquée dans le
            # checkpoint (compatible préset mac / tailles personnalisées)
            _, self.diffusion, self.image_size = load_ddpm_from_ckpt(
                ckpt, self.device)
        if cgan_ckpt:
            cfg = Config()
            self.cgan = Generator(cfg.gan.latent_dim, image_size=image_size,
                                  base=cfg.gan.base_channels).to(self.device)
            ck = torch.load(cgan_ckpt, map_location=self.device)
            self.cgan.load_state_dict(ck["G"])
            self.cgan.eval()

    def _age_norm(self, age):
        """Âge voulu (années) -> valeur de condition normalisée, calibrée."""
        if self.calibration is not None:
            age = float(self.calibration.years([age])[0])
        return normalize_age(age)

    def _attrs(self, age, gender, skin, n):
        return {
            "age": torch.full((n,), self._age_norm(age), device=self.device),
            "gender": torch.full((n,), gender, dtype=torch.long,
                                 device=self.device),
            "skin": torch.full((n,), skin, dtype=torch.long,
                               device=self.device),
        }

    def _placeholder(self, n, text="Mode démo\nEntraînez le modèle"):
        """Images factices pour tester l'UI sans checkpoint."""
        rng = np.random.default_rng(0)
        return [(rng.uniform(20, 60, (self.image_size, self.image_size, 3))
                 .astype(np.uint8)) for _ in range(n)]

    @staticmethod
    def _to_pil_list(x):
        arr = ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1) \
            .cpu().numpy()
        return [a for a in arr]

    def generate(self, age, gender, skin, n, guidance, steps, seed,
                 progress_cb=None):
        if self.demo:
            return self._placeholder(n)
        if seed >= 0:
            torch.manual_seed(seed)
        attrs = self._attrs(age, gender, skin, n)

        def gen(a, m):
            return self.diffusion.sample_ddim(
                a, (m, 3, self.image_size, self.image_size),
                steps=int(steps), guidance_scale=guidance,
                progress_cb=progress_cb)

        if self.best_of > 1 and self.clf is not None:
            x = sample_best_of(gen, attrs, n, self.clf, k=self.best_of)
        else:
            x = gen(attrs, n)
        if self.reject_artifacts:
            x = resample_artifacts(gen, attrs, n, x=x)
        return self._to_pil_list(x)

    def interpolate(self, age_a, gender_a, skin_a, age_b, gender_b, skin_b,
                    frames, guidance, seed):
        if self.demo:
            return self._placeholder(frames)

        def mk(age, g, s):
            return {"age": torch.tensor([self._age_norm(age)]),
                    "gender": torch.tensor([g]), "skin": torch.tensor([s])}
        x = interp_fn(self.diffusion, mk(age_a, gender_a, skin_a),
                      mk(age_b, gender_b, skin_b), frames=int(frames),
                      image_size=self.image_size, guidance_scale=guidance,
                      seed=int(seed) if seed >= 0 else 0)
        return self._to_pil_list(x)

    def make_gif(self, age_a, gender_a, skin_a, age_b, gender_b, skin_b,
                 frames, guidance, seed, fps=8):
        """Interpolation exportée en GIF animé aller-retour (boomerang)."""
        import tempfile
        from PIL import Image as PILImage
        imgs = self.interpolate(age_a, gender_a, skin_a, age_b, gender_b,
                                skin_b, frames, guidance, seed)
        pil = [PILImage.fromarray(np.asarray(a)).resize(
            (self.image_size * 4, self.image_size * 4), PILImage.NEAREST)
            for a in imgs]
        pil = pil + pil[-2:0:-1]  # boomerang
        path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
        pil[0].save(path, save_all=True, append_images=pil[1:],
                    duration=int(1000 / fps), loop=0)
        return path, path

    def atlas(self, ages, guidance, steps, seed, progress_cb=None):
        """Atlas démographique : MÊME identité (même bruit initial) déclinée
        sur 7 groupes de peau x 2 genres x âges choisis."""
        ages = sorted(ages) or [45]
        combos = [(s, g, a) for s in range(len(SKIN_GROUPS))
                  for g in (0, 1) for a in ages]
        n = len(combos)
        captions = [f"{SKIN_CHOICES[s]} · {'H' if g == 0 else 'F'} · {a} ans"
                    for s, g, a in combos]
        if self.demo:
            return list(zip(self._placeholder(n), captions))
        g_rng = torch.Generator(device=self.device).manual_seed(int(seed))
        x_T = torch.randn(1, 3, self.image_size, self.image_size,
                          device=self.device, generator=g_rng) \
            .repeat(n, 1, 1, 1)
        attrs = {
            "age": torch.tensor([self._age_norm(a) for _, _, a in combos],
                                device=self.device),
            "gender": torch.tensor([g for _, g, _ in combos],
                                   dtype=torch.long, device=self.device),
            "skin": torch.tensor([s for s, _, _ in combos],
                                 dtype=torch.long, device=self.device),
        }
        x = self.diffusion.sample_ddim(
            attrs, (n, 3, self.image_size, self.image_size),
            steps=int(steps), guidance_scale=guidance, x_T=x_T,
            progress_cb=progress_cb)
        return list(zip(self._to_pil_list(x), captions))

    def compare(self, age, gender, skin, n, guidance, seed,
                progress_cb=None):
        ddpm = self.generate(age, gender, skin, n, guidance, 50, seed,
                             progress_cb=progress_cb)
        if self.cgan is None:
            return ddpm, self._placeholder(n)
        if seed >= 0:
            torch.manual_seed(seed)
        attrs = self._attrs(age, gender, skin, n)
        with torch.no_grad():
            z = torch.randn(n, Config().gan.latent_dim, device=self.device)
            x = self.cgan(z, attrs)
        return ddpm, self._to_pil_list(x)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
GRADIO_MAJOR = int(gr.__version__.split(".")[0])


def build_ui(engine):
    # Sur CPU (déploiement Hugging Face Spaces gratuit), un visage coûte ~8 s à
    # 30 pas DDIM : on réduit les valeurs par défaut pour que la première
    # génération réponde en quelques secondes plutôt qu'en deux minutes.
    on_cpu = str(getattr(engine, "device", "cpu")) == "cpu"
    n_default, steps_default = (4, 30) if on_cpu else (8, 50)
    theme = gr.themes.Base(
        primary_hue="violet", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ).set(
        body_background_fill="#07080f",
        block_background_fill="rgba(255,255,255,0.035)",
        block_border_color="rgba(255,255,255,0.09)",
        input_background_fill="rgba(255,255,255,0.05)",
    )

    def attr_controls(prefix=""):
        age = gr.Slider(18, 70, value=30, step=1,
                        label=f"Âge {prefix}".strip(), interactive=True)
        gender = gr.Radio(choices=[("Homme", 0), ("Femme", 1)], value=0,
                          label=f"Genre {prefix}".strip())
        skin = gr.Dropdown(choices=[(c, i) for i, c in enumerate(SKIN_CHOICES)],
                           value=0, label=f"Tonalité de peau {prefix}".strip())
        return age, gender, skin

    # Gradio >= 6 : theme/css passent à launch() ; < 6 : au constructeur
    blocks_kw = {} if GRADIO_MAJOR >= 6 else {"theme": theme, "css": CSS}
    with gr.Blocks(title="VISAGE", **blocks_kw) as demo:
        gr.HTML("""
        <div id="hero">
          <h1>VISAGE</h1>
          <p class="sub">Génération conditionnelle de visages photo-réalistes</p>
          <div class="badges">
            <span class="badge"><b>DDPM</b> conditionnel</span>
            <span class="badge">Classifier-Free <b>Guidance</b></span>
            <span class="badge"><b>DDIM</b> 50 pas</span>
            <span class="badge">PyTorch <b>from scratch</b></span>
            <span class="badge">Dataset <b>FairFace</b></span>
          </div>
        </div>""")

        # ------------------ Onglet 1 : Génération ------------------
        with gr.Tab("✦ Génération"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        age, gender, skin = attr_controls()
                    with gr.Accordion("Paramètres avancés", open=False):
                        n = gr.Slider(1, 16, value=n_default, step=1,
                                      label="Nombre de visages")
                        guidance = gr.Slider(1.0, 8.0, value=3.0, step=0.5,
                                             label="Guidance (fidélité ↔ diversité)")
                        steps = gr.Slider(10, 200, value=steps_default, step=10,
                                          label="Pas DDIM (vitesse ↔ qualité)")
                        seed = gr.Number(value=-1, label="Seed (-1 = aléatoire)",
                                         precision=0)
                    btn = gr.Button("Générer", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Visages générés", columns=4,
                                         height=560, object_fit="cover")
            btn.click(engine.generate,
                      [age, gender, skin, n, guidance, steps, seed], gallery)

        # ------------------ Onglet 2 : Interpolation ------------------
        with gr.Tab("⇄ Interpolation"):
            gr.Markdown(
                "Même bruit initial (identité fixe), attributs interpolés "
                "continûment dans l'espace d'embedding : le visage se "
                "transforme progressivement.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Point de départ A**")
                    age_a, gender_a, skin_a = attr_controls("A")
                with gr.Column():
                    gr.Markdown("**Point d'arrivée B**")
                    age_b, gender_b, skin_b = attr_controls("B")
            with gr.Row():
                frames = gr.Slider(4, 16, value=8, step=1, label="Étapes")
                guid_i = gr.Slider(1.0, 8.0, value=3.0, step=0.5,
                                   label="Guidance")
                seed_i = gr.Number(value=0, label="Seed identité", precision=0)
            btn_i = gr.Button("Interpoler", variant="primary", size="lg")
            strip = gr.Gallery(label="Transition A → B", columns=8, height=220,
                               object_fit="cover")
            btn_i.click(engine.interpolate,
                        [age_a, gender_a, skin_a, age_b, gender_b, skin_b,
                         frames, guid_i, seed_i], strip)
            with gr.Row():
                btn_gif = gr.Button("◉ Exporter en GIF animé", size="lg")
                gif_view = gr.Image(label="Animation (boomerang)", height=280)
                gif_file = gr.File(label="Télécharger le GIF")
            btn_gif.click(engine.make_gif,
                          [age_a, gender_a, skin_a, age_b, gender_b, skin_b,
                           frames, guid_i, seed_i], [gif_view, gif_file])

        # ------------------ Onglet 3 : Atlas ------------------
        with gr.Tab("◈ Atlas démographique"):
            gr.Markdown(
                "**Une seule identité** (bruit initial fixé), déclinée sur les "
                "7 groupes de peau × 2 genres × âges choisis. Vérification "
                "visuelle directe : le contrôle modifie les attributs "
                "démographiques sans changer la « personne ».")
            with gr.Row():
                ages_sel = gr.CheckboxGroup(
                    choices=[25, 45, 65], value=[25, 45, 65],
                    label="Âges de l'atlas")
                guid_at = gr.Slider(1.0, 8.0, value=3.0, step=0.5,
                                    label="Guidance")
                steps_at = gr.Slider(10, 100, value=30, step=10,
                                     label="Pas DDIM")
                seed_at = gr.Number(value=7, label="Seed identité",
                                    precision=0)
            btn_at = gr.Button("Générer l'atlas", variant="primary", size="lg")
            atlas_gal = gr.Gallery(label="Atlas (survolez pour les attributs)",
                                   columns=6, height=640, object_fit="cover")
            btn_at.click(engine.atlas, [ages_sel, guid_at, steps_at, seed_at],
                         atlas_gal)

        # ------------------ Onglet 4 : DDPM vs cGAN ------------------
        with gr.Tab("⚖ Comparaison DDPM / cGAN"):
            gr.Markdown("Mêmes attributs, deux familles de modèles : diffusion "
                        "conditionnelle (notre approche) vs GAN conditionnel "
                        "(baseline).")
            with gr.Row():
                age_c, gender_c, skin_c = attr_controls()
                with gr.Column():
                    n_c = gr.Slider(1, 8, value=4, step=1, label="Visages")
                    guid_c = gr.Slider(1.0, 8.0, value=3.0, step=0.5,
                                       label="Guidance (DDPM)")
                    seed_c = gr.Number(value=42, label="Seed", precision=0)
            btn_c = gr.Button("Comparer", variant="primary", size="lg")
            with gr.Row():
                gal_ddpm = gr.Gallery(label="DDPM + CFG", columns=4, height=280)
                gal_cgan = gr.Gallery(label="cGAN (baseline)", columns=4,
                                      height=280)
            btn_c.click(engine.compare,
                        [age_c, gender_c, skin_c, n_c, guid_c, seed_c],
                        [gal_ddpm, gal_cgan])

        # ------------------ Onglet 4 : À propos ------------------
        with gr.Tab("ℹ À propos"):
            gr.Markdown(f"""
### Pipeline
`Attributs (âge, genre, peau)` → `Embeddings appris` → `UNet conditionnel`
→ `DDIM {50} pas + Classifier-Free Guidance` → `Visage 64×64`

### Contrôles
| Attribut | Encodage | Plage |
|---|---|---|
| Âge | scalaire continu → embedding sinusoïdal | 18 – 70 ans |
| Genre | embedding de classe | Homme / Femme |
| Peau | embedding de classe | 7 groupes (FairFace) |

### Guidance (w)
`ε = ε_uncond + w · (ε_cond − ε_uncond)` - augmente la fidélité aux
attributs au prix de la diversité. w = 3 est un bon compromis.

### Éthique
Modèle entraîné sur **FairFace**, conçu pour l'équilibre démographique.
Les visages générés sont synthétiques : aucune personne réelle n'est
représentée. Usage réservé à la recherche/pédagogie ;
tout usage d'usurpation d'identité est proscrit.

*Mode : {"DÉMO (aucun checkpoint chargé)" if engine.demo else "modèle chargé ✓"} -
device : {engine.device}*
""")
    demo._visage_theme, demo._visage_css = theme, CSS
    return demo


if __name__ == "__main__":
    import os
    p = argparse.ArgumentParser()
    # Variables d'environnement = déploiement Hugging Face Spaces
    # (Spaces lance `python app.py` sans arguments)
    p.add_argument("--ckpt", default=os.environ.get("FACEGEN_CKPT"),
                   help="checkpoint DDPM (.pt)")
    p.add_argument("--cgan-ckpt", default=os.environ.get("FACEGEN_CGAN_CKPT"),
                   help="checkpoint cGAN (.pt)")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--demo", action="store_true", help="UI sans modèle")
    p.add_argument("--share", action="store_true")
    p.add_argument("--no-calibration", action="store_true",
                   help="désactive la calibration post-hoc de l'âge")
    p.add_argument("--best-of", type=int, default=1,
                   help="k candidats par requête, le plus conforme est affiché "
                        "(nécessite --clf) ; coût x k")
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    args = p.parse_args()
    # Fallback : si un ckpt est présent aux emplacements standards, le charger
    if args.ckpt is None and not args.demo:
        for cand in ("ckpt_last.pt", "runs/ddpm/ckpt_last.pt"):
            if os.path.exists(cand):
                args.ckpt = cand
                break

    engine = Engine(ckpt=args.ckpt, cgan_ckpt=args.cgan_ckpt,
                    image_size=args.image_size, demo=args.demo,
                    calibration=None if args.no_calibration else "auto",
                    best_of=args.best_of, clf=args.clf)
    ui = build_ui(engine)
    launch_kw = {"share": args.share}
    if GRADIO_MAJOR >= 6:
        launch_kw.update(theme=ui._visage_theme, css=ui._visage_css)
    ui.launch(**launch_kw)
