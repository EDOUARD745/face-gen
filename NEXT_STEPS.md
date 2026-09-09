# Suite de l'entraînement — instructions prêtes à copier-coller

> **Machine** : Apple M5 Pro (MPS, 24 Go). Interpréteur = `python3` (`/usr/local/bin/python3`,
> Framework 3.13, où torch+MPS et toutes les dépendances sont installés).
> **Ne PAS utiliser `python`** (= anaconda, sans les dépendances).
>
> ⚠️ **Toujours préfixer** les commandes d'entraînement/éval par `PYTORCH_ENABLE_MPS_FALLBACK=1`.
>
> ⚠️ **Toujours ajouter `PYTHONUNBUFFERED=1`** (ou `python3 -u`) pour les runs redirigés vers un
> fichier : sinon les `print()` restent bloqués dans le buffer stdout et `tail -f` **ne montre rien
> pendant très longtemps** (le run tourne pourtant). Alternative de suivi fiable : TensorBoard.

---

## a) Suivi du cGAN (en cours, lancé détaché)

```bash
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen"

# Logs en direct (grâce à PYTHONUNBUFFERED=1) : pertes D/G toutes les 100 steps
tail -f logs/cgan.log

# Grilles de contrôle générées toutes les 1000 steps (qualité + respect des attributs)
open runs/cgan/grid_*.png            # la plus récente

# TensorBoard (courbes loss_d / loss_g + échantillons)
tensorboard --logdir runs

# Le process tourne-t-il encore ?
pgrep -fl train_cgan

# Reprise du cGAN si interrompu (checkpoint toutes les 5000 steps + fin d'epoch)
caffeinate -i nohup env PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 python3 -m src.train_cgan \
    --data-root data/fairface --image-size 48 --epochs 60 --resume runs/cgan/ckpt_last.pt \
    > logs/cgan.log 2>&1 &
```

Repère de qualité : `loss_d` et `loss_g` doivent rester bornées (pas de divergence vers ±∞).
Un GAN baseline reste plus instable qu'un DDPM : surveiller les grilles.

---

## b) DDPM — À LANCER *UNIQUEMENT APRÈS LA FIN DU cGAN* (jamais les deux en parallèle : contention GPU)

Vérifier d'abord que le cGAN est terminé : `pgrep -fl train_cgan` ne doit **rien** renvoyer.

```bash
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen"

# Lancement (détaché, ~18 h pour 40 epochs à ~39 img/s — voir rapport)
caffeinate -i nohup env PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 python3 -m src.train_ddpm \
    --data-root data/fairface --preset mac > logs/ddpm.log 2>&1 &

# Suivi
tail -f logs/ddpm.log                # loss toutes les 100 steps
open runs/ddpm/grid_*.png            # grilles toutes les 2000 steps (7 peaux x 2 genres x 3 âges)
tensorboard --logdir runs

# Reprise après interruption (checkpoint toutes les 5000 steps + fin d'epoch)
caffeinate -i nohup env PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1 python3 -m src.train_ddpm \
    --resume runs/ddpm/ckpt_last.pt > logs/ddpm.log 2>&1 &
```

### ⚠️ Risque NaN (IMPORTANT — observé au smoke test)
Lors des tests, la loss DDPM descendait bien (0.065 au step 100 → **0.040 au step 300**) puis, sur
**un run sur trois**, est partie en **NaN vers le step 800** (instabilité numérique fp32 sur MPS).
Sur un run de 80 000 steps, ce risque n'est pas négligeable et **`ckpt_last.pt` écrase les bons poids
par des poids NaN** une fois la divergence installée.

Recommandations :
- **Surveiller** `tail -f logs/ddpm.log` : si des `nan` apparaissent, **arrêter immédiatement**
  (`pkill -f train_ddpm`) — inutile de laisser tourner.
- En cas de NaN récurrent, relancer avec un **learning rate plus faible** (flag CLI, aucune modif de code) :
  ```bash
  ... python3 -m src.train_ddpm --data-root data/fairface --preset mac --lr 1e-4 > logs/ddpm.log 2>&1 &
  ```
- Sauvegarder périodiquement un checkpoint sain à part si un bon état est atteint :
  `cp runs/ddpm/ckpt_last.pt runs/ddpm/ckpt_ok_$(date +%s).pt` (ne jamais supprimer un checkpoint).

---

## c) Évaluation — À LANCER UNE FOIS LE DDPM TERMINÉ

Toutes en `--image-size 48` (résolution commune DDPM/cGAN) et `PYTORCH_ENABLE_MPS_FALLBACK=1`.
Séquentiel (une à la fois).

```bash
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen"

# 1. Métriques DDPM (FID, IS, LPIPS intra-condition, fidélité attributs)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.evaluate --model ddpm \
    --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 -n 5000

# 2. Métriques cGAN (baseline)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.evaluate --model cgan \
    --ckpt runs/cgan/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 -n 5000

# 3. Compromis fidélité/diversité du CFG (figures pour le rapport)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.sweep_guidance \
    --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 --scales 1 2 3 5 8 -n 2000

# 4. Équité démographique : FID + fidélité PAR groupe de peau
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.eval_by_group \
    --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
    --data-root data/fairface --image-size 48 -n 1000

# 5. Interpolation continue d'âge (figure)
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m src.interpolate \
    --ckpt runs/ddpm/ckpt_last.pt --age-a 20 --age-b 68 --frames 8 --out figures/interp_age.png
```

## d) Démonstrateur (interface Gradio)

```bash
# --image-size 48 OBLIGATOIRE : le checkpoint cGAN n'embarque pas sa taille et
# a été entraîné en 48px (le DDPM, lui, se reconstruit depuis sa config).
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 app.py \
    --ckpt runs/ddpm/ckpt_last.pt --cgan-ckpt runs/cgan/ckpt_last.pt --image-size 48
# puis ouvrir http://127.0.0.1:7860
```

---

## Rappels d'état (au moment de la rédaction)
- ✅ Données FairFace prêtes : `data/fairface/` (64 599 train / 8 100 val après filtre 18-70 ans).
- ✅ Classifieur entraîné : `runs/classifier/attr_clf.pt` (genre 92.9 %, peau 62.1 %, MAE âge 6.5 ans).
- 🔄 cGAN 48px : en cours (détaché).
- ⏳ DDPM : à lancer après le cGAN (voir §b, attention au risque NaN).

## Correctifs de code appliqués (voir rapport)
- `src/models/cgan.py` : support de la résolution **48px** (amorce 3×3 ; le DCGAN ne produisait que
  64/128px). Nécessaire car la mission impose 48px pour la parité DDPM/cGAN. 64/128px non régressés.
- `src/evaluate.py`, `src/sweep_guidance.py`, `src/eval_by_group.py` : **FID/IS de torchmetrics
  gardés sur CPU** (ils accumulent en float64, incompatible MPS → `TypeError: Cannot convert a MPS
  Tensor to float64`). Le sampling, LPIPS et le classifieur restent sur MPS ; seul le calcul
  Inception du FID passe sur CPU. Aucun impact sur les valeurs, juste un peu plus lent.
