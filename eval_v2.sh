#!/bin/bash
# Évaluation complète du modèle v2 (fine-tune age-jitter + masquage indépendant).
# Séquentiel : un seul job GPU à la fois.
cd "$(dirname "$0")" || exit 1
PY=/usr/local/bin/python3
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
CK=runs/ddpm_ft2/ckpt_last.pt
CLF=runs/classifier/attr_clf.pt
COMMON="--clf $CLF --data-root data/fairface --image-size 48"

echo "=== 1/6 Réponse en âge, CFG standard w=3 ==="
$PY -m src.age_response --ckpt $CK --clf $CLF --image-size 48 -n 48 --step 4 \
    --out-png figures/age_response_v2.png --out-csv figures/age_response_v2.csv \
    --out-json runs/ddpm_ft2/age_calibration.json

echo "=== 2/6 Réponse en âge, guidage compositionnel w_age=10 ==="
$PY -m src.age_response --ckpt $CK --clf $CLF --image-size 48 -n 48 --step 4 \
    --guidance-scale 3 --guidance-age 10 \
    --out-png figures/age_response_v2_comp.png \
    --out-csv figures/age_response_v2_comp.csv \
    --out-json runs/ddpm_ft2/age_calibration_comp.json

echo "=== 3/6 Métriques v2, CFG standard (protocole identique à v1) ==="
$PY -m src.evaluate --model ddpm --ckpt $CK $COMMON -n 5000

echo "=== 4/6 Métriques v2, guidage compositionnel w_age=10 ==="
$PY -m src.evaluate --model ddpm --ckpt $CK $COMMON -n 2000 --guidance-age 10

echo "=== 5/6 Équité démographique v2 ==="
$PY -m src.eval_by_group --ckpt $CK $COMMON -n 1000

echo "=== 6/6 Interpolation d'âge (figure) ==="
$PY -m src.interpolate --ckpt $CK --age-a 20 --age-b 68 --frames 8 \
    --out figures/interp_age_v2.png

echo "=== TERMINÉ ==="
