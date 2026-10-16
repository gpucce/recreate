#!/usr/bin/env bash
#
# coco_dog (1000 COCO objects, images in data_bkp/coco/) with the qwen38-27b_flux2klein models: same experiment,
# GPU layout and env vars as vlstereoset_qwen38-27b_flux2klein.sh, only the dataset changes. Output goes to
# data_bkp/coco_dog_experiments/qwen38-27b_flux2klein/<run name>/.
#
#   experiment_scripts/coco_dog_qwen38-27b_flux2klein.sh --dry-run    # plan only
#   experiment_scripts/coco_dog_qwen38-27b_flux2klein.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/coco_dog_qwen38-27b_flux2klein.sh   # only GPU 2
#   experiment_scripts/coco_dog_qwen38-27b_flux2klein.sh --status | --stop

set -euo pipefail
export DATASET="coco_dog"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_qwen38-27b_flux2klein.sh" "$@"
