#!/usr/bin/env bash
set -euo pipefail

# Remove generated image/comparison files under outputs/ while preserving
# checkpoints, logs, JSON/NPZ artifacts, model weights, and images produced
# during training.
#
# Usage:
#   tools/clean_outputs_images.sh           # delete image files
#   tools/clean_outputs_images.sh --dry-run # show files only

ROOT_DIR="outputs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Directory '$ROOT_DIR' does not exist. Nothing to do."
  exit 0
fi

# Image formats commonly produced by experiment/generation pipelines.
# Exclusions preserve training-time visual outputs.
FIND_EXPR=(
  -type f
  \( \
    -iname "*.png" -o \
    -iname "*.jpg" -o \
    -iname "*.jpeg" -o \
    -iname "*.tif" -o \
    -iname "*.tiff" -o \
    -iname "*.webp" \
  \)
  ! -path "*/training_visualizations/*"
  ! -path "*/tensorboard_logs/*"
  ! -path "*/[0-9]/real_x_generated_facies/*"
  ! -path "*/[0-9]/real_x_generated_rock_physics/*"
  ! -path "*/[0-9]/real_x_generated_ip/*"
  ! -path "*/[0-9]/real_x_generated_is/*"
)

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] Files that would be removed from '$ROOT_DIR':"
  find "$ROOT_DIR" "${FIND_EXPR[@]}" | sort
  COUNT=$(find "$ROOT_DIR" "${FIND_EXPR[@]}" | wc -l)
  echo "[dry-run] Total files: $COUNT"
  exit 0
fi

COUNT=$(find "$ROOT_DIR" "${FIND_EXPR[@]}" | wc -l)
if [[ "$COUNT" -eq 0 ]]; then
  echo "No image files found under '$ROOT_DIR'."
  exit 0
fi

echo "Removing $COUNT image files under '$ROOT_DIR'..."
find "$ROOT_DIR" "${FIND_EXPR[@]}" -print -delete

echo "Done."
