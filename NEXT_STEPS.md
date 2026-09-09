# Suite du projet — instructions prêtes à copier-coller

> **Machine** : Apple M5 Pro (MPS, 24 Go). Interpréteur = **`/usr/local/bin/python3`**
> (Framework 3.13 : torch+MPS, tensorboard, lpips, torchmetrics y sont installés).
> **Ne PAS utiliser `python3` nu** : le PATH le résout vers anaconda, qui n'a pas
> tensorboard (l'entraînement s'arrête à l'import).
>
> ⚠️ **Toujours préfixer** par `PYTORCH_ENABLE_MPS_FALLBACK=1`.
> ⚠️ **Toujours ajouter `PYTHONUNBUFFERED=1`** pour les runs redirigés vers un fichier,
> sinon `tail -f` ne montre rien pendant très longtemps.
> ⚠️ **Un seul job GPU à la fois** (contention MPS) : ne pas évaluer pendant un entraînement.

---

## État au dernier point

- ✅ Données FairFace : `data/fairface/` (64 599 train / 8 100 val avec les points
  médians ; **73 702** avec `--age-jitter`, voir plus bas).
- ✅ Classifieur : `runs/classifier/attr_clf.pt` (genre 92,9 %, peau 62,1 %, MAE âge 6,5 ans).
- ✅ cGAN 48 px : `runs/cgan/ckpt_last.pt` (FID 139,6 — mode collapse conditionnel).
- ✅ DDPM v1 : `runs/ddpm/ckpt_last.pt`, 40 epochs / 80 720 steps (FID 40,2).
- ✅ Évaluations v1 : FID/IS/LPIPS/fidélité, sweep CFG, équité par groupe, interpolation.
- ⚠️ **Défaut identifié** : le contrôle d'âge est inopérant (réponse plate, amplitude
  7,7 ans) — cause = support d'âge réduit à 5 valeurs par le mapping tranche→médian
  suivi du filtre 18-70. Diagnostic : `figures/age_response.png`.
- 🔄 **DDPM v2 (fine-tune `--age-jitter`)** : en cours dans `runs/ddpm_ft/`.
- ✅ Dépôt git initialisé (données et poids exclus).

---

## a) Fine-tune du contrôle d'âge (DDPM v2)

```bash
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen"

caffeinate -i nohup env PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 \
    /usr/local/bin/python3 -m src.train_ddpm --data-root data/fairface \
    --preset mac --age-jitter --resume runs/ddpm/ckpt_last.pt \
    --batch-size 32 --lr 5e-5 --epochs 48 --out-dir runs/ddpm_ft \
    > logs/ddpm_ft.log 2>&1 &

tail -f logs/ddpm_ft.log        # loss toutes les 100 steps
open runs/ddpm_ft/grid_*.png    # grilles toutes les 2000 steps
pgrep -fl train_ddpm            # tourne toujours ?
```

`--out-dir runs/ddpm_ft` : le modèle v1 n'est jamais écrasé, la comparaison
avant/après reste possible. Reprendre après interruption : remplacer
`--resume runs/ddpm/ckpt_last.pt` par `--resume runs/ddpm_ft/ckpt_last.pt`
(l'architecture est relue depuis le checkpoint, `--preset` n'est plus obligatoire).

⚠️ **Risque NaN** (observé sur le run v1, instabilité fp32 sur MPS) : surveiller
`tail -f logs/ddpm_ft.log`, arrêter (`pkill -f train_ddpm`) si des `nan` apparaissent,
et relancer avec un lr plus faible. `ckpt_prev.pt` conserve l'avant-dernier état sain.

## b) Vérifier le gain de contrôle d'âge (après le fine-tune)

```bash
# Courbe de réponse du modèle affiné -> figure avant/après pour le rapport
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.age_response \
    --ckpt runs/ddpm_ft/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --image-size 48 -n 48 --out-png figures/age_response_ft.png \
    --out-csv figures/age_response_ft.csv \
    --out-json runs/ddpm_ft/age_calibration.json
```

Lecture : la courbe doit se rapprocher de la diagonale ; l'amplitude de réponse
(imprimée en fin de script) doit passer de 7,7 ans à plusieurs dizaines d'années.

## c) Ré-évaluation complète du modèle v2

```bash
# Métriques principales (FID, IS, LPIPS, fidélité)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.evaluate --model ddpm \
    --ckpt runs/ddpm_ft/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 -n 5000

# Fidélité seule, rapide, avec échantillonnage par rejet
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.evaluate --model ddpm \
    --ckpt runs/ddpm_ft/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 --skip-fid --best-of 4

# Sweep CFG et équité par groupe (comme pour v1, pour comparer)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.sweep_guidance \
    --ckpt runs/ddpm_ft/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 --scales 1 2 3 5 8 -n 2000
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.eval_by_group \
    --ckpt runs/ddpm_ft/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 -n 1000

# Interpolation d'âge (figure) — c'est la démonstration visuelle du gain
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 -m src.interpolate \
    --ckpt runs/ddpm_ft/ckpt_last.pt --age-a 20 --age-b 68 --frames 8 \
    --out figures/interp_age_ft.png
```

## d) Démonstrateur

```bash
# VISAGE Studio (soutenance)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 server.py \
    --ckpt runs/ddpm_ft/ckpt_last.pt        # -> http://localhost:8000

# Gradio (fallback / HF Spaces). --image-size 48 obligatoire pour le cGAN.
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 app.py \
    --ckpt runs/ddpm_ft/ckpt_last.pt --cgan-ckpt runs/cgan/ckpt_last.pt \
    --image-size 48

# Option : sélection best-of-4 par le classifieur (x4 plus lent, plus fidèle)
PYTORCH_ENABLE_MPS_FALLBACK=1 /usr/local/bin/python3 app.py \
    --ckpt runs/ddpm_ft/ckpt_last.pt --image-size 48 --best-of 4
```

---

## Correctifs de code appliqués

- `src/data.py` : mode `--age-jitter` — l'âge est tiré dans la tranche FairFace
  annotée (bornée à 18-70) au lieu du point médian. Restaure un support continu
  et récupère les 18-19 ans (9 103 images). Comportement historique conservé par
  défaut pour la reproductibilité du run v1.
- `src/train_ddpm.py` : `--resume` relit l'architecture depuis la config embarquée
  dans le checkpoint (avant, une reprise sans `--preset` reconstruisait un UNet
  64 px/128 canaux et échouait au `load_state_dict`) ; nouveau flag `--age-jitter`.
- `src/age_response.py` (nouveau) : mesure de la fonction de réponse en âge,
  figure, et table de calibration marquée `applicable: false` quand la réponse
  est trop plate pour être inversée.
- `src/sampling.py` (nouveau) : calibration post-hoc de la condition d'âge et
  échantillonnage par rejet best-of-k guidé par le classifieur.
- `src/evaluate.py` : `--age-sampling {uniform,support}`, `--calibration`,
  `--best-of`, `--skip-fid`, `--fidelity-n`.
- `app.py` / `server.py` : calibration appliquée à toutes les conditions d'âge
  (génération, morphose, atlas) et option `--best-of`.
- `src/models/cgan.py` : support 48 px (amorce 3×3).
- `src/evaluate.py`, `src/sweep_guidance.py`, `src/eval_by_group.py` : FID/IS de
  torchmetrics gardés sur CPU (accumulation float64 incompatible MPS).
