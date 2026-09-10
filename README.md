# VISAGE : Génération de visages photo-réalistes avec contrôle d'attributs

Projet MSC AIC. Système de génération conditionnelle de visages contrôlant
**l'âge (18-70 ans, continu)**, **le genre** et **la couleur de peau**
(7 groupes FairFace).

**Approche principale** : DDPM conditionnel entraîné from scratch +
Classifier-Free Guidance, échantillonnage DDIM (quasi temps réel).
**Baseline de comparaison** : cGAN à discriminateur à projection.

**Modèle livré : v2** (`runs/ddpm_ft2/ckpt_last.pt`). FID 24,0, MAE d'âge
10,8 ans, fidélité genre 86,8 %. Il succède à v1 (FID 40,2) dont le contrôle
d'âge était inopérant ; le diagnostic et la correction sont documentés
ci-dessous et au §6-8 du rapport d'expériences.

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
│   └── Etat_de_l_art.docx  # Livrable 1 (30%) : ouvrir dans Word,
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
# 1. Classifieur d'attributs (~1h) : nécessaire pour l'évaluation de fidélité
python -m src.train_classifier --data-root data/fairface --epochs 10

# 2. Baseline cGAN (quelques heures)
python -m src.train_cgan --data-root data/fairface --image-size 64

# 3. DDPM conditionnel (modèle principal) : ~24-48h sur RTX 3080/4090 en 64px
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

## Contrôle de l'âge : défaut identifié, diagnostic et correction

**Le défaut.** FairFace n'annote pas l'âge à l'année mais par tranches. Le
chargeur les ramenait au point médian *puis* filtrait sur 18-70 ans : après ce
filtre il ne restait que **cinq valeurs d'âge distinctes** (24,5 / 34,5 / 44,5 /
54,5 / 64,5) et la tranche 10-19 disparaissait entièrement, alors qu'elle couvre
les 18-19 ans exigés par le sujet. Le modèle n'a donc jamais vu d'exemple sous
24,5 ans ni au-dessus de 64,5 ans, et le protocole d'évaluation lui demandait
des âges hors de ce support.

**Le diagnostic** (`src/age_response.py`) mesure la fonction de réponse
« âge perçu = f(âge demandé) », le classifieur d'attributs servant de juge :

```bash
python -m src.age_response --ckpt runs/ddpm/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --image-size 48 -n 48
```

Sorties : `figures/age_response.png` (figure du rapport), `age_response.csv`,
et une table de calibration `runs/ddpm/age_calibration.json`. Sur le premier
modèle, la réponse mesurée est **plate** (amplitude 7,7 ans pour une demande
allant de 18 à 70 ans) : le contrôle d'âge est absent, pas seulement imprécis.
La calibration est alors automatiquement marquée `applicable: false` et ignorée
à l'inférence : inverser une fonction plate reviendrait à maquiller l'absence
de contrôle.

**La correction** restaure un support continu : l'âge est tiré uniformément
dans la tranche annotée, bornée à 18-70 (`--age-jitter`, cf. `src/data.py`).
Le dataset passe de 64 599 images / 5 âges à **73 702 images / âges continus**.
Le modèle est repris depuis le checkpoint existant (aucune perte des 18 h déjà
investies) et affiné :

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.train_ddpm \
    --data-root data/fairface --preset mac --age-jitter \
    --resume runs/ddpm/ckpt_last.pt --batch-size 32 --lr 5e-5 \
    --epochs 48 --out-dir runs/ddpm_ft
```

L'écriture dans un répertoire distinct garantit que le modèle initial reste
intact et que la comparaison avant/après est reproductible.

**Mitigations à l'inférence** (`src/sampling.py`, sans ré-entraînement) :
calibration de la condition (quand la réponse est inversible) et échantillonnage
par rejet : k candidats générés, le plus conforme au sens du classifieur est
retenu (Azadi et al., 2019) :

```bash
# Fidélité seule (rapide), avec sélection best-of-4
python -m src.evaluate --model ddpm --ckpt runs/ddpm/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface \
    --image-size 48 --skip-fid --best-of 4
```

Le démonstrateur expose la même option (`app.py --best-of 4`, `server.py
--best-of 4`) ; le coût d'échantillonnage est multiplié par k.

**Protocole d'évaluation.** `--age-sampling support` tire les conditions d'âge
dans le support réellement appris, `uniform` (défaut) sur 18-70. Les deux sont
rapportés : le premier mesure le contrôle, le second l'écart au cahier des
charges.

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

**Option A, lien temporaire (zéro config)** :
`python app.py --ckpt ... --share` → URL publique `xxx.gradio.live`,
valable tant que la machine tourne (72h max par lien). Idéal soutenance.

**Option B, lien permanent sur Hugging Face Spaces**

⚠️ **Depuis 2025, un Space Gradio ou Docker sur `cpu-basic` exige un compte
PRO** (9 $/mois). L'API renvoie sinon `402 Payment Required`. Seuls les
**Static Spaces** (HTML/JS, sans backend Python) restent gratuits.

Le bundle de déploiement est préparé dans `deploy/space/` (61 Mo) :

```
deploy/space/
├── app.py             # interface Gradio (défauts abaissés si device == cpu)
├── src/               # modèle, diffusion, échantillonnage
├── ckpt_last.pt       # poids EMA seuls : 64 Mo au lieu de 254
├── requirements.txt   # inférence uniquement (ni tensorboard, ni lpips…)
└── README.md          # en-tête YAML du Space + avertissements + éthique
```

Le régénérer après un nouvel entraînement :

```bash
python3 -c "
import torch; ck = torch.load('runs/ddpm_ft2/ckpt_last.pt', map_location='cpu')
torch.save({'ema': ck['ema'], 'config': ck['config'], 'step': ck['step'],
            'epoch': ck['epoch']}, 'deploy/space/ckpt_last.pt')"
```

Le pousser (compte PRO requis, token en écriture) :

```bash
python3 -c "
from huggingface_hub import create_repo, upload_folder
R = 'elouamou/visage-generation-visages'
create_repo(R, repo_type='space', space_sdk='gradio', private=False, exist_ok=True)
upload_folder(repo_id=R, repo_type='space', folder_path='deploy/space')"
```

**Performances CPU mesurées** (2 threads, équivalent `cpu-basic`) :
7,8 s par visage à 30 pas DDIM, 13,6 s à 50 pas. Un Space endormi
(48 h sans visite) met 30 à 60 s à se réveiller : le signaler dans le
README du Space, sinon un visiteur pressé conclut à une panne.

Ne pas pousser le dataset FairFace (licence, et inutile à la démo) : seul
le checkpoint sert.

**Si le budget PRO n'est pas disponible** : un Static Space gratuit peut
héberger une galerie pré-calculée (grille de visages couvrant l'espace
d'attributs, générée hors ligne). Instantané pour le visiteur, mais ce
n'est plus le modèle qui tourne, à écrire explicitement sur la page.

## Reproductibilité

Seed fixée (`config.py`), configuration sauvegardée en JSON dans chaque run,
checkpoints périodiques avec état optimiseur, poids EMA utilisés pour toute
génération/évaluation. Le code est versionné sous git (`git log`) ; les
données FairFace et les poids sont exclus du dépôt (licence, volume).

Une reprise (`--resume`) relit l'architecture depuis la configuration
embarquée dans le checkpoint : elle fonctionne sans avoir à repasser
`--preset`.

## Tests

```bash
python3 -m tests.test_pipeline
```

Vérifications sans GPU ni checkpoint : réversibilité de l'encodage d'âge,
cohérence tranches/points médians, **support d'âge du dataset** (dégénéré en
mode historique, continu avec `--age-jitter`), refus d'une calibration non
inversible, et sélection best-of-k. C'est l'absence de ce dernier type de
vérification qui a laissé un support d'âge à cinq valeurs traverser 18 h
d'entraînement.

## Fidélité colorimétrique et filtre du démonstrateur

Deux problèmes distincts, dont un seul se corrige.

**Biais systématique d'éclaircissement.** Le modèle éclaircit la peau de 0,053
en luminance moyenne, et d'autant plus qu'elle est foncée : +0,095 pour le
groupe Black (0,323 réel contre 0,418 généré), +0,034 pour White. L'étendue
entre groupes se comprime de 32 % (0,088 à 0,060). Aucun post-traitement n'est
appliqué : ramener les couleurs vers les statistiques du groupe demandé
fabriquerait la fidélité de l'attribut que le projet mesure.

**Tirages aberrants.** Le démonstrateur écarte et régénère ceux dont un canal
s'éloigne de plus de 3,21 écarts-types des statistiques des visages réels
(99,5e centile). Le taux atteint 27 à 29 %. Ni les pas de débruitage (30, 50,
80) ni le guidage (1,5 à 3) ne le réduisent. Le filtre est réservé au
démonstrateur et désactivable ; **aucune métrique du rapport ne l'utilise.**

## Limites connues

* **Résolution 48 px** : imposée par le budget de calcul (entraînement sur
  machine personnelle, MPS). Le terme « photo-réaliste » du sujet n'est atteint
  qu'au sens du réalisme distributionnel mesuré (FID) à cette résolution ; les
  FID absolus ne sont pas comparables aux références haute résolution de la
  littérature. Seules les comparaisons internes, à protocole constant, sont
  interprétables.
* **Montée en résolution tentée, puis abandonnée** : un entraînement en 96 px
  repris depuis v2 (8,3 epochs) améliore nettement l'image : netteté 76 % du
  réel contre 46 % pour v2 agrandi, mais **perd le contrôle d'âge** (MAE
  14,0 ans contre 6,1). Huit epochs sur quatorze n'ont pas suffi à réinstaller
  le conditionnement à la nouvelle échelle. Voir rapport §9 ; checkpoint
  conservé dans `runs/ddpm_96/`, figures dans `figures/age_response_96.*`.
* **Contrôle d'âge** : inopérant sur v1, rétabli sur v2 (`runs/ddpm_ft2/`).
  Effectif entre ~26 et ~62 ans ; aux extrémités (18 et 70 ans) la réponse
  reste tirée vers le centre, faute de données. Entre 26 et 62 ans, l'erreur
  passe sous celle du juge sur images réelles (6,5 ans) : la métrique sature.
* **Budget de conditionnement** : les trois attributs sont sommés en un
  vecteur unique, donc renforcer le guidage de l'âge dégrade le genre et la
  peau (~3,5 points de fidélité genre par année de MAE gagnée, cf.
  `figures/age_guidance_tradeoff.png`). Le réglage livré est le guidage
  standard, sans supplément d'âge.
* **Juge d'évaluation** : le classifieur de fidélité plafonne à 62 % sur la
  peau et commet 6,5 ans d'erreur sur images réelles ; les scores de fidélité
  sont donc des bornes inférieures.

## Éthique (résumé, détails dans le rapport, section 7)

Dataset FairFace choisi pour son équilibre démographique ; évaluation par
groupe et non en moyenne seule ; visages 100 % synthétiques (aucune
inversion/édition de personnes réelles possible) ; usage strictement
pédagogique, usurpation d'identité proscrite.
