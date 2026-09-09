#!/bin/bash
cd "$(dirname "$0")" || exit 1
PY=/usr/local/bin/python3
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
CK=runs/ddpm_ft2/ckpt_last.pt
for wa in 0 2 4 6 9; do
  echo "===== supplément d'âge w_age=$wa ====="
  if [ "$wa" = "0" ]; then
    $PY -m src.evaluate --model ddpm --ckpt $CK --clf runs/classifier/attr_clf.pt \
        --data-root data/fairface --image-size 48 --skip-fid --fidelity-n 500
  else
    $PY -m src.evaluate --model ddpm --ckpt $CK --clf runs/classifier/attr_clf.pt \
        --data-root data/fairface --image-size 48 --skip-fid --fidelity-n 500 \
        --guidance-mode hybrid --guidance-age $wa
  fi
done
echo "===== SWEEP TERMINÉ ====="
