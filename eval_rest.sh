#!/bin/zsh
# Reprise des évaluations restantes (3, 4, 5) — les 1 et 2 sont déjà faites.
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
mkdir -p figures

echo "[auto2] $(date '+%H:%M:%S') reprise des évals restantes"

echo "[auto2] EVAL 3/5 : sweep guidance (5 échelles × 2000)..."
python3 -m src.sweep_guidance --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 --scales 1 2 3 5 8 -n 2000 > logs/eval_sweep.log 2>&1 \
  && echo "[auto2] EVAL 3/5 OK -> logs/eval_sweep.log + figures/" || echo "[auto2] EVAL 3/5 ÉCHEC -> logs/eval_sweep.log"

echo "[auto2] EVAL 4/5 : équité par groupe (n=1000)..."
python3 -m src.eval_by_group --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 -n 1000 > logs/eval_group.log 2>&1 \
  && echo "[auto2] EVAL 4/5 OK -> logs/eval_group.log + figures/" || echo "[auto2] EVAL 4/5 ÉCHEC -> logs/eval_group.log"

echo "[auto2] EVAL 5/5 : interpolation d'âge..."
python3 -m src.interpolate --ckpt runs/ddpm/ckpt_last.pt --age-a 20 --age-b 68 --frames 8 \
  --out figures/interp_age.png > logs/eval_interp.log 2>&1 \
  && echo "[auto2] EVAL 5/5 OK -> figures/interp_age.png" || echo "[auto2] EVAL 5/5 ÉCHEC -> logs/eval_interp.log"

echo "[auto2] $(date '+%H:%M:%S') ÉVALS RESTANTES TERMINÉES. Voir logs/eval_*.log et figures/"
