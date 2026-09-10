# VISAGE : génération de visages avec contrôle d'attributs démographiques

Projet MSC AIC, module IA générative.
Edouard Louamou, Isaac Koumous, Jeffrey Tandjeu, Kelian Suami.

Système de génération conditionnelle de visages permettant de contrôler
l'âge (18 à 70 ans, continu), le genre et la tonalité de peau (7 groupes
FairFace). L'approche principale est un modèle de diffusion débruitante
(DDPM) conditionnel entraîné depuis zéro, avec classifier-free guidance et
échantillonnage DDIM. Un cGAN à discriminateur à projection sert de méthode
de référence, entraîné sur les mêmes données, à la même résolution et avec
les mêmes encodages d'attributs.

L'ensemble a été entraîné sur une machine personnelle (Apple M5 Pro, backend
Metal), sans GPU serveur. Cette contrainte a dicté la résolution de travail
de 48 pixels et le dimensionnement des modèles.

```
Attributs            Embeddings          UNet conditionnel      DDIM 50 pas
(âge, genre, peau) → appris           →  (ε-prédicteur,      →  + guidage    → Visage
                     (sinusoïdal +       15,9 M paramètres)     w réglable
                      tables)
```

## Résultats

Modèle livré : `runs/ddpm_ft2/ckpt_last.pt` (48 epochs cumulées, 99 144 steps).

| Métrique | DDPM v2 | DDPM v1 | cGAN | Lecture |
|---|---|---|---|---|
| FID ↓ | **24,0** | 40,2 | 139,6 | réalisme distributionnel |
| Inception Score ↑ | 3,31 | 3,35 | 2,01 | qualité et variété |
| Diversité LPIPS ↑ | 0,388 | 0,401 | 0,001 | 0 = effondrement des modes |
| Fidélité genre ↑ | 86,8 % | 89,6 % | 70,4 % | juge : classifieur ResNet-18 |
| Fidélité peau ↑ | 44,7 % | 45,8 % | 21,1 % | plafond du juge : 62 % |
| MAE d'âge ↓ | 10,8 ans | 12,5 ans | 12,2 ans | 6,1 ans avec guidage d'âge |

Le juge est un classifieur tri-têtes entraîné séparément sur FairFace
(genre 92,9 %, peau 62,1 %, MAE d'âge 6,5 ans en validation). Les fidélités
rapportées sont donc des bornes inférieures : l'erreur du générateur et celle
de l'instrument de mesure s'y additionnent.

Deux résultats méthodologiques sont détaillés dans le rapport d'expériences
et résumés plus bas : un défaut de conditionnement de l'âge, diagnostiqué et
corrigé (section 6 à 8 du rapport), et un biais colorimétrique mesuré mais
délibérément non corrigé (section 8.1).

## Structure du dépôt

```
face-gen/
├── app.py                  # démonstrateur Gradio
├── server.py               # démonstrateur VISAGE Studio (FastAPI)
├── studio/index.html       # interface du Studio
├── rapport/                # état de l'art et rapport d'expériences (.docx)
├── figures/                # figures et données des rapports
├── deploy/space/           # bundle de déploiement Hugging Face
├── tests/test_pipeline.py  # tests de non-régression
└── src/
    ├── config.py           # hyperparamètres centralisés
    ├── data.py             # FairFace, UTKFace, encodage des attributs
    ├── diffusion.py        # DDPM, cosine schedule, DDIM, CFG, EMA
    ├── models/unet.py      # UNet conditionnel
    ├── models/cgan.py      # baseline cGAN
    ├── sampling.py         # calibration d'âge, rejet, best-of-k
    ├── train_ddpm.py       # entraînement du modèle principal
    ├── train_cgan.py       # entraînement de la baseline
    ├── train_classifier.py # juge d'évaluation
    ├── evaluate.py         # FID, IS, LPIPS, fidélité aux attributs
    ├── age_response.py     # fonction de réponse en âge
    ├── compare_age_response.py  # figure comparative
    ├── grid_stats.py       # suivi de qualité sans coût GPU
    ├── sweep_guidance.py   # compromis fidélité/diversité du CFG
    ├── eval_by_group.py    # métriques par groupe démographique
    └── interpolate.py      # interpolation continue d'attributs
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Le device est détecté automatiquement dans l'ordre CUDA, MPS, CPU.

## Données

FairFace, images « padding 0.25 », à télécharger depuis
https://github.com/joojs/fairface et organiser ainsi :

```
data/fairface/
├── fairface_label_train.csv
├── fairface_label_val.csv
├── train/*.jpg
└── val/*.jpg
```

FairFace annote l'âge par tranches. Deux modes de lecture coexistent :

- **mode historique** : chaque tranche est ramenée à son point médian, puis le
  filtre 18-70 ans est appliqué. 64 599 images d'entraînement, mais cinq
  valeurs d'âge distinctes seulement.
- **mode `--age-jitter`** : l'âge est tiré uniformément dans la tranche
  annotée, bornée à 18-70 ans. 73 702 images et un support continu.

Le second est celui du modèle livré. Le premier est conservé pour reproduire
le run initial. UTKFace est également pris en charge (`--dataset utkface`),
avec un âge annoté à l'année.

## Entraînement

Ordre conseillé : classifieur, puis cGAN, puis DDPM.

```bash
# 1. Juge d'évaluation (~1 h)
python -m src.train_classifier --data-root data/fairface --epochs 10

# 2. Baseline cGAN
python -m src.train_cgan --data-root data/fairface --image-size 48

# 3. DDPM conditionnel, configuration du modèle livré
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m src.train_ddpm \
    --data-root data/fairface --preset mac --age-jitter --independent-drop \
    --ema-decay 0.999 --batch-size 32 --lr 1e-4 --epochs 48
```

Les présets `mac`, `mac64` et `mac96` fixent la résolution à 48, 64 ou 96
pixels. L'attention est placée à résolution/4, donc toujours au troisième
étage du UNet : la structure des modules reste identique d'une résolution à
l'autre, ce qui permet de reprendre un checkpoint 48 px pour poursuivre
l'entraînement à une résolution supérieure.

Une reprise (`--resume`) relit l'architecture depuis la configuration
embarquée dans le checkpoint. Sur MPS, une instabilité numérique en fp32 peut
provoquer une divergence vers NaN ; `ckpt_prev.pt` conserve l'avant-dernier
état et un pas d'apprentissage plus faible résout le problème.

## Évaluation

```bash
# Métriques principales
python -m src.evaluate --model ddpm --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface \
    --image-size 48 -n 5000

# Compromis fidélité/diversité du guidage
python -m src.sweep_guidance --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface \
    --image-size 48 --scales 1 2 3 5 8 -n 2000

# Équité démographique
python -m src.eval_by_group --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --data-root data/fairface \
    --image-size 48 -n 1000

# Fonction de réponse en âge
python -m src.age_response --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --clf runs/classifier/attr_clf.pt --image-size 48 -n 48
```

L'option `--age-sampling` choisit la distribution des conditions d'âge :
`uniform` tire sur 18-70 ans, `support` restreint aux valeurs effectivement
présentes dans les données. Les deux sont rapportées, l'écart entre elles
mesurant le coût de l'extrapolation.

En haute résolution, le lot d'évaluation est plafonné et le cache
d'allocation libéré à chaque lot : sans cette précaution, une évaluation en
96 px sature la mémoire unifiée.

## Démonstrateur

Deux interfaces partagent le même moteur.

```bash
# VISAGE Studio, interface sur mesure (FastAPI), recommandée en soutenance
python server.py --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --cgan-ckpt runs/cgan/ckpt_last.pt --image-size 48

# Gradio, également utilisée pour le déploiement en ligne
python app.py --ckpt runs/ddpm_ft2/ckpt_last.pt \
    --cgan-ckpt runs/cgan/ckpt_last.pt --image-size 48
```

Cinq onglets : génération conditionnelle, interpolation continue à identité
fixe, atlas démographique (une identité déclinée sur 7 peaux × 2 genres ×
âges choisis), comparaison DDPM contre cGAN à attributs identiques, et une
note technique et éthique.

Latences mesurées, guidage actif : 1,31 s par visage en 50 pas DDIM sur MPS,
0,53 s par visage pour un lot de six en 30 pas, 7,8 s sur processeur seul.

### Déploiement en ligne

Le bundle prêt à publier se trouve dans `deploy/space/` (61 Mo : poids EMA
seuls, calibration d'âge, générateur du cGAN, interface, dépendances
d'inférence). Depuis 2025, héberger un Space Gradio requiert un abonnement
PRO pour un compte personnel, ou un plan Team pour une organisation.

```bash
python -c "
from huggingface_hub import upload_folder
upload_folder(repo_id='<compte>/<space>', repo_type='space',
              folder_path='deploy/space')"
```

## Deux résultats méthodologiques

### Contrôle de l'âge : un défaut invisible aux métriques agrégées

Le premier modèle affichait des métriques globales acceptables tout en
ignorant la condition d'âge. La mesure de la fonction de réponse, c'est-à-dire
l'âge perçu sur les visages produits en fonction de l'âge demandé, montre une
courbe plate : amplitude de 7,7 ans pour une consigne balayant 52 ans.

La cause n'était pas dans le modèle mais dans le chargeur de données. Le
passage par les points médians, suivi du filtre 18-70 ans, réduisait le
support d'entraînement à cinq valeurs et supprimait la tranche 10-19, donc
les 18 et 19 ans exigés par le sujet.

Trois correctifs, dont aucun ne suffit seul :

| Configuration | Amplitude de réponse | MAE moyenne |
|---|---|---|
| v1 : support à 5 valeurs, masquage groupé | 7,7 ans | 14,3 ans |
| v2 : support continu, masquage indépendant | 16,2 ans | 11,6 ans |
| v2 avec guidage d'âge compositionnel | 29,7 ans | 6,1 ans |

Le masquage indépendant par attribut est le correctif décisif. Les trois
attributs étant sommés puis masqués en bloc par le CFG, le réseau pouvait
annuler sa perte en s'appuyant sur le genre et la peau sans jamais exploiter
l'âge. Le guidage compositionnel s'obtient au prix d'une dégradation des
autres attributs, mesurée dans le rapport : environ 3,5 points de fidélité au
genre par année de MAE gagnée. Le réglage livré est donc le guidage standard.

### Fidélité colorimétrique : un biais mesuré et non corrigé

Le modèle éclaircit systématiquement la peau, de 0,053 en luminance moyenne,
et d'autant plus qu'elle est foncée : +0,095 pour le groupe Black contre
+0,034 pour White. L'étendue de luminance entre groupes se comprime de 32 %.

Aucun post-traitement n'est appliqué. Ramener les couleurs vers les
statistiques du groupe demandé produirait une fidélité artificielle sur
l'attribut même que le projet évalue.

Le démonstrateur écarte en revanche les tirages franchement aberrants, ceux
dont un canal s'éloigne de plus de 3,21 écarts-types des statistiques des
visages réels, soit leur 99,5e centile. Ce filtre concerne 27 à 29 % des
tirages selon la tonalité demandée. Il est réservé à la démonstration et
désactivable : aucune métrique de ce dépôt ne l'utilise.

## Limites connues

- **Résolution de 48 pixels**, imposée par le budget de calcul. Les FID
  absolus ne sont pas comparables aux références haute résolution de la
  littérature ; seules les comparaisons internes, à protocole constant, le
  sont. Une tentative de montée à 96 px améliore l'image (netteté à 76 % du
  réel contre 46 % pour v2 agrandi) mais perd le contrôle d'âge après
  8 epochs sur les 14 prévues. Checkpoint conservé dans `runs/ddpm_96/`.
- **Contrôle d'âge effectif entre 26 et 62 ans environ.** Aux extrémités, la
  réponse reste tirée vers le centre, faute de données. Dans cette plage,
  l'erreur passe sous celle du juge sur images réelles : la métrique sature.
- **Équité.** L'affinage améliore le FID de tous les groupes (de 60-75 à
  41-57) sans modifier leur hiérarchie. Il relève le niveau général, il ne
  corrige pas l'inégalité entre groupes.
- **Budget de conditionnement.** Les trois attributs transitent par un
  vecteur unique injecté en un point du réseau : renforcer l'un dégrade les
  autres. Un conditionnement par cross-attention lèverait cette contrainte.

## Reproductibilité et tests

Graine fixée dans `config.py`, configuration sauvegardée en JSON dans chaque
run, checkpoints périodiques avec état de l'optimiseur, poids EMA utilisés
pour toute génération et toute évaluation. Le code est versionné sous git ;
les données FairFace et les poids sont exclus du dépôt pour des raisons de
licence et de volume.

```bash
python3 -m tests.test_pipeline
```

Les tests vérifient sans GPU ni checkpoint : la réversibilité de l'encodage
d'âge, la cohérence entre tranches et points médians, le support d'âge du
jeu de données dans les deux modes, le refus d'une calibration non
inversible, la sélection best-of-k, et les deux régimes de masquage du CFG.
L'absence de ce dernier type de vérification est ce qui a laissé un support
d'âge dégénéré traverser dix-huit heures d'entraînement.

## Éthique

FairFace a été retenu pour son équilibre démographique. Les visages produits
sont entièrement synthétiques : le modèle génère depuis du bruit et ne peut
ni reconstituer ni éditer une personne réelle. L'évaluation est menée par
groupe démographique et non en moyenne seule, ce qui rend visibles des écarts
qu'un agrégat masquerait, notamment le biais d'éclaircissement documenté plus
haut. L'usage est strictement pédagogique ; toute usurpation d'identité est
proscrite. La réflexion complète figure en section 7 de l'état de l'art.
