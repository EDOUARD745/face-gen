#!/bin/zsh
# Orchestrateur auto : surveille le DDPM (NaN), puis lance l'évaluation complète.
# Lancé détaché sous `nohup caffeinate -dimsu` -> survit à la fermeture de session
# ET maintient l'écran allumé (pas de verrouillage) pendant toute la durée.
cd "/Users/edouardlouamou/Documents/Projets_Perso/Projet IA Generative/face-gen" || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

echo "[auto] $(date '+%H:%M:%S') surveillance DDPM démarrée"

# --- A) Attendre la fin du DDPM en surveillant le NaN ---
while pgrep -f train_ddpm >/dev/null 2>&1; do
  if grep -qiE "loss nan|nan \(" logs/ddpm.log 2>/dev/null; then
    pkill -f train_ddpm
    echo "[auto] $(date '+%H:%M:%S') NaN DÉTECTÉ — DDPM arrêté. Évaluation ANNULÉE."
    echo "[auto] Relancer le DDPM avec --lr 1e-4 (voir NEXT_STEPS.md §b)."
    exit 0
  fi
  sleep 120
done

# --- B) Vérifier une terminaison PROPRE avant d'évaluer ---
if grep -qiE "loss nan" logs/ddpm.log 2>/dev/null; then
  echo "[auto] $(date '+%H:%M:%S') DDPM terminé mais NaN présent — évaluation annulée."; exit 0
fi
if [ ! -f runs/ddpm/ckpt_last.pt ]; then
  echo "[auto] $(date '+%H:%M:%S') checkpoint DDPM absent — évaluation annulée."; exit 0
fi
if ! grep -q "Entra" logs/ddpm.log 2>/dev/null; then
  echo "[auto] $(date '+%H:%M:%S') DDPM arrêté avant la fin (pas de message de fin) — évaluation NON lancée par sécurité. Vérifier logs/ddpm.log."; exit 0
fi

echo "[auto] $(date '+%H:%M:%S') DDPM terminé proprement — lancement de l'évaluation"
mkdir -p figures

# --- C) Suite d'évaluation séquentielle (une à la fois, tolérante aux échecs) ---
echo "[auto] EVAL 1/5 : métriques DDPM (FID/IS/LPIPS/fidélité)..."
python3 -m src.evaluate --model ddpm --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 -n 5000 > logs/eval_ddpm.log 2>&1 \
  && echo "[auto] EVAL 1/5 OK -> logs/eval_ddpm.log" || echo "[auto] EVAL 1/5 ÉCHEC -> logs/eval_ddpm.log"

echo "[auto] EVAL 2/5 : métriques cGAN (baseline)..."
python3 -m src.evaluate --model cgan --ckpt runs/cgan/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 -n 5000 > logs/eval_cgan.log 2>&1 \
  && echo "[auto] EVAL 2/5 OK -> logs/eval_cgan.log" || echo "[auto] EVAL 2/5 ÉCHEC -> logs/eval_cgan.log"

echo "[auto] EVAL 3/5 : sweep guidance (courbes CFG)..."
python3 -m src.sweep_guidance --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 --scales 1 2 3 5 8 -n 2000 > logs/eval_sweep.log 2>&1 \
  && echo "[auto] EVAL 3/5 OK -> logs/eval_sweep.log" || echo "[auto] EVAL 3/5 ÉCHEC -> logs/eval_sweep.log"

echo "[auto] EVAL 4/5 : équité par groupe démographique..."
python3 -m src.eval_by_group --ckpt runs/ddpm/ckpt_last.pt --clf runs/classifier/attr_clf.pt \
  --data-root data/fairface --image-size 48 -n 1000 > logs/eval_group.log 2>&1 \
  && echo "[auto] EVAL 4/5 OK -> logs/eval_group.log" || echo "[auto] EVAL 4/5 ÉCHEC -> logs/eval_group.log"

echo "[auto] EVAL 5/5 : interpolation d'âge (figure)..."
python3 -m src.interpolate --ckpt runs/ddpm/ckpt_last.pt --age-a 20 --age-b 68 --frames 8 \
  --out figures/interp_age.png > logs/eval_interp.log 2>&1 \
  && echo "[auto] EVAL 5/5 OK -> figures/interp_age.png" || echo "[auto] EVAL 5/5 ÉCHEC -> logs/eval_interp.log"

echo "[auto] $(date '+%H:%M:%S') ÉVALUATION COMPLÈTE. Résultats dans logs/eval_*.log et figures/"
