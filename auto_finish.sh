#!/bin/bash
# Arrêt programmé de l'entraînement 96px puis évaluation complète, sans
# intervention. Conçu pour tourner pendant une absence.
#
#   1. attend l'heure d'arrêt (STOP_AT)
#   2. arrête l'entraînement — le dernier checkpoint a au plus ~1 h
#   3. exporte les poids EMA seuls (bundle de déploiement)
#   4. mesure : réponse en âge, FID/IS/LPIPS/fidélité, interpolation
#
# Tout est journalisé dans logs/auto_finish.log. Rien n'écrase v2 :
# le modèle 48px reste dans runs/ddpm_ft2/.
cd "$(dirname "$0")" || exit 1
PY=/usr/local/bin/python3
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
STOP_AT="${1:-15:30}"
CK=runs/ddpm_96/ckpt_last.pt
CLF=runs/classifier/attr_clf.pt

echo "=== arrêt programmé à $STOP_AT (maintenant $(date +%H:%M)) ==="
while [ "$(date +%H:%M)" != "$STOP_AT" ]; do
  pgrep -f "train_ddpm.*ddpm_96" >/dev/null || { echo "entraînement déjà terminé à $(date +%H:%M)"; break; }
  sleep 20
done

if pgrep -f "train_ddpm.*ddpm_96" >/dev/null; then
  echo "=== arrêt de l'entraînement à $(date +%H:%M) ==="
  pkill -f "train_ddpm.*ddpm_96"
  sleep 20
fi
echo "dernier checkpoint : $(stat -f '%Sm' -t '%H:%M' $CK)"
$PY -m src.grid_stats runs/ddpm_96 --image-size 96 --last 8

echo "=== 1/4 réponse en âge (96px, guidage standard) ==="
$PY -m src.age_response --ckpt $CK --clf $CLF --image-size 96 -n 32 --step 4 \
    --out-png figures/age_response_96.png --out-csv figures/age_response_96.csv \
    --out-json runs/ddpm_96/age_calibration.json

echo "=== 2/4 métriques principales (96px) ==="
$PY -m src.evaluate --model ddpm --ckpt $CK --clf $CLF \
    --data-root data/fairface --image-size 96 -n 2000 --fidelity-n 500

echo "=== 3/4 interpolation d'âge (figure) ==="
$PY -m src.interpolate --ckpt $CK --age-a 20 --age-b 68 --frames 8 \
    --out figures/interp_age_96.png

echo "=== 4/4 bundle de déploiement (poids EMA seuls) ==="
$PY - <<'PYEOF'
import torch, os, pathlib
ck = torch.load("runs/ddpm_96/ckpt_last.pt", map_location="cpu")
pathlib.Path("deploy/space").mkdir(parents=True, exist_ok=True)
torch.save({"ema": ck["ema"], "config": ck["config"],
            "step": ck["step"], "epoch": ck["epoch"]},
           "deploy/space/ckpt_96.pt")
print(f"deploy/space/ckpt_96.pt : {os.path.getsize('deploy/space/ckpt_96.pt')/1e6:.0f} Mo "
      f"(step {ck['step']}, epoch {ck['epoch']})")
PYEOF

echo "=== TERMINÉ à $(date +%H:%M) ==="
