# VISAGE — Génération de visages photo-réalistes avec contrôle d'attributs

Projet MSC AIC. Système de génération conditionnelle de visages contrôlant
**l'âge (18-70 ans, continu)**, **le genre** et **la couleur de peau**
(7 groupes FairFace).

**Approche principale** : DDPM conditionnel entraîné from scratch +
Classifier-Free Guidance, échantillonnage DDIM (quasi temps réel).
**Baseline de comparaison** : cGAN à discriminateur à projection.

```
Attributs ──► Embeddings appris ──► UNet conditionnel ──► DDIM 50 pas + CFG ──► Visage
 (âge, genre,    (sinusoïdal +         (ε-prédicteur,        w réglable
  peau)           tables)               ~75M params)
```

## Structure

```
face-gen/
├── app.py                  # Interface Gradio (démonstrateur)
├── requirements.txt
├── rapport/
│   └── Etat_de_l_art.docx  # Livrable 1 (30%) — ouvrir dans Word,
│                           #   accepter la mise à jour des champs (sommaire)
└── src/
    ├── config.py           # Hyperparamètres centralisés (reproductibilité)
    ├── data.py             # FairFace / UTKFace + encodage des attributs
    ├── diffusion.py        # DDPM, cosine schedule, DDIM, CFG, EMA
    ├── models/
    │   ├── unet.py         # UNet conditionnel (temps + attributs)
    │   └── cgan.py         # Baseline cGAN (projection discriminator)
    ├── train_ddpm.py       # Entraînement diffusion
    ├── train_cgan.py       # Entraînement baseline
    ├── train_classifier.py # Classifieur d'attributs (pour l'évaluation)
    ├── evaluate.py         # FID, IS, LPIPS intra-condition, fidélité attributs
    ├── sweep_guidance.py   # Courbes FID/fidélité/diversité vs guidance w
    ├── eval_by_group.py    # Métriques PAR groupe démographique (équité)
    └── interpolate.py      # Interpolation continue d'attributs
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Données (FairFace)

1. Télécharger depuis https://github.com/joojs/fairface :
   images "padding 0.25" (train + val) et les deux CSV de labels.
2. Organiser :

```
data/fairface/
├── fairface_label_train.csv
├── fairface_label_val.csv
├── train/*.jpg
└── val/*.jpg
```

Le loader filtre automatiquement les âges hors 18-70 ans (~78k images restantes).
Alternative : UTKFace (`--dataset utkface`, âge exact à l'année).

## Entraînement

Ordre conseillé : classifieur (rapide, requis pour l'évaluation) → cGAN
(valide le pipeline données à moindre coût) → DDPM (le gros morceau).

```bash
# 1. Classifieur d'attributs (~1h) — nécessaire pour l'évaluation de fidélité
python -m src.train_classifier --data-root data/fairface --epochs 10

# 2. Baseline cGAN (quelques heures)
python -m src.train_cgan --data-root data/fairface --image-size 64

# 3. DDPM conditionnel (modèle principal) — ~24-48h sur RTX 3080/4090 en 64px
python -m src.train_ddpm --data-root data/fairface --image-size 64 \
    --batch-size 64 --epochs 100

# Reprise après interruption
python -m src.train_ddpm --resume runs/ddpm/ckpt_last.pt

# Suivi : tensorboard --logdir runs
```

**Avant de lancer les 48h** : laisser tourner ~15 min et vérifier que
(a) la loss DDPM descend nettement sous 0.1, (b) pas d'erreur mémoire
(sinon `--batch-size 32`). Pendant l'entraînement, surveiller les grilles
`runs/ddpm/grid_*.png` : dès 10-20k steps, des visages flous doivent se
structurer et chaque colonne respecter ses attributs.

Des grilles de contrôle (7 peaux × 2 genres × 3 âges) sont générées
périodiquement dans `runs/ddpm/` pour suivre visuellement la qualité
ET le respect des attributs.

Conseils GPU : si mémoire insuffisante, réduire `--batch-size` (32) ;
`base_channels=96` dans `src/config.py` divise le modèle par ~1.8.

**Matériel** : le device est détecté automatiquement (CUDA > MPS > CPU).

### Entraînement sur Mac Apple Silicon (MPS)

Le préset `mac` allège le modèle (16M params, 48px) pour ramener le DDPM
à ~1-2 jours sur M1/M2/M3 :

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train_ddpm \
    --data-root data/fairface --preset mac
```

La configuration est embarquée dans les checkpoints : app, évaluation et
interpolation reconstruisent automatiquement la bonne architecture
(`app.py --ckpt ...` fonctionne tel quel). Pour les scripts d'évaluation,
passer `--image-size 48` afin que les images réelles soient comparées à la
même résolution. Le classifieur et le cGAN peuvent rester en réglages par
défaut (rapides même sur MPS). Alternative si les délais sont trop longs :
Google Colab / Kaggle (30h GPU/semaine gratuites).

## Évaluation

```bash
python -m src.evaluate --model ddpm --ckpt runs/ddpm/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface -n 5000
python -m src.evaluate --model cgan --ckpt runs/cgan/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface -n 5000
```

Métriques produites : **FID** (réalisme distributionnel), **IS**,
**diversité LPIPS intra-condition** (détecte le mode collapse conditionnel),
**fidélité aux attributs** (accord genre/peau %, MAE âge en années).
Deux analyses supplémentaires produisent des **figures prêtes pour le rapport**
(`figures/*.png` + données CSV) :

```bash
# Compromis fidélité/diversité du CFG (courbes FID, fidélité, LPIPS vs w)
python -m src.sweep_guidance --ckpt runs/ddpm/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface \
    --scales 1 2 3 5 8 -n 2000

# Équité démographique : FID + fidélité PAR groupe de peau
python -m src.eval_by_group --ckpt runs/ddpm/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface -n 1000
```

## Interfaces (démonstrateur)

Deux interfaces branchées sur le même moteur :

**VISAGE Studio (recommandée pour la soutenance)** - frontend web sur mesure
(FastAPI + HTML/JS), design éditorial : pastilles de peau colorées, âge en
chiffre géant, progression du débruitage en direct, historique de séance,
onglets Studio / Morphose / Atlas.

```bash
python server.py --ckpt runs/ddpm/ckpt_last.pt   # -> http://localhost:8000
python server.py --demo                          # design seul, sans modèle
```

**Gradio (fallback + déploiement HF Spaces)** :

```bash
python app.py --ckpt runs/ddpm/ckpt_last.pt --cgan-ckpt runs/cgan/ckpt_last.pt
# Sans modèle entraîné (test de l'UI seule) :
python app.py --demo
```

5 onglets : **Génération** (attributs + guidance + seed), **Interpolation**
(identité fixe, attributs continûment variés + **export GIF animé** du
vieillissement), **Atlas démographique** (une identité déclinée sur
7 peaux × 2 genres × 3 âges), **Comparaison DDPM/cGAN** (mêmes attributs,
deux modèles), **À propos** (pipeline + éthique).

## Déploiement public (lien cliquable pour l'évaluation)

**Option A — lien temporaire (zéro config)** :
`python app.py --ckpt ... --share` → URL publique `xxx.gradio.live`,
valable tant que la machine tourne (72h max par lien). Idéal soutenance.

**Option B — lien permanent : Hugging Face Spaces (recommandé pour le rapport)**

1. Créer un compte sur https://huggingface.co → New Space → SDK **Gradio**,
   hardware **CPU basic** (gratuit).
2. Pousser dans le Space : `app.py`, `src/`, `requirements.txt`
   et le checkpoint `ckpt_last.pt` (à la racine — l'app le détecte
   automatiquement ; fichier >10 Mo : `git lfs track "*.pt"` avant commit).
3. Le Space se construit et sert l'app à une URL permanente
   `https://huggingface.co/spaces/<user>/<space>`.

Sur CPU gratuit, compter ~10-30 s par visage avec le modèle préset mac
(48px, DDIM 30 pas) — réduire le slider "Pas DDIM" dans l'interface.
Ne pas pousser le dataset FairFace (licence + poids inutile) : seul le
checkpoint est nécessaire à la démo.

## Reproductibilité

Seed fixée (`config.py`), configuration sauvegardée en JSON dans chaque run,
checkpoints périodiques avec état optimiseur, poids EMA utilisés pour toute
génération/évaluation.

## Éthique (résumé — détails dans le rapport, section 7)

Dataset FairFace choisi pour son équilibre démographique ; évaluation par
groupe et non en moyenne seule ; visages 100 % synthétiques (aucune
inversion/édition de personnes réelles possible) ; usage strictement
pédagogique, usurpation d'identité proscrite.
