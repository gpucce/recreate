#!/usr/bin/env bash
#
# coco_car (1000 COCO objects, images in data_bkp/coco/) with the qwen38-27b_zimage models: same experiment,
# GPU layout and env vars as vlstereoset_qwen38-27b_zimage.sh, only the dataset changes. Output goes to
# data_bkp/coco_car_experiments/qwen38-27b_zimage/<run name>/.
#
#   experiment_scripts/coco_car_qwen38-27b_zimage.sh --dry-run    # plan only
#   experiment_scripts/coco_car_qwen38-27b_zimage.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/coco_car_qwen38-27b_zimage.sh   # only GPU 2
#   experiment_scripts/coco_car_qwen38-27b_zimage.sh --status | --stop

set -euo pipefail
export DATASET="coco_car"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_qwen38-27b_zimage.sh" "$@"
