#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MANIFEST="${MANIFEST:-../boundingBoxes.json}"
IMAGES="${IMAGES:-../images}"
MASKS_DIR="${MASKS_DIR:-masks}"
OUT="${OUT:-predictions.json}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
CHECKPOINT="${CHECKPOINT:-models/sam_vit_b_01ec64.pth}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && ! grep -q '^OPENROUTER_API_KEY=' .env 2>/dev/null; then
  echo "Missing OpenRouter API key. Set OPENROUTER_API_KEY in .env or export it before running." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision segment-anything opencv-python pillow numpy

if [ ! -f "$CHECKPOINT" ]; then
  mkdir -p "$(dirname "$CHECKPOINT")"
  curl -L --fail -o "$CHECKPOINT" "$CHECKPOINT_URL"
fi

rm -rf "$MASKS_DIR"
.venv/bin/python scripts/sam_masks.py \
  --manifest "$MANIFEST" \
  --images "$IMAGES" \
  --checkpoint "$CHECKPOINT" \
  --model-type "$MODEL_TYPE" \
  --out "$MASKS_DIR"

cargo run --release -- \
  --manifest "$MANIFEST" \
  --images "$IMAGES" \
  --masks "$MASKS_DIR" \
  --out "$OUT"

echo "Wrote $OUT"
