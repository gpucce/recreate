#!/usr/bin/env bash
#
# coco_dog (1000 COCO objects, images in data_bkp/coco/) with the qwen3vl4b_zimage models: same experiment,
# GPU layout and env vars as vlstereoset_qwen3vl4b_zimage.sh, only the dataset changes. Output goes to
# data_bkp/coco_dog_experiments/qwen3vl4b_zimage/<run name>/.
#
#   experiment_scripts/coco_dog_qwen3vl4b_zimage.sh --dry-run    # plan only
#   experiment_scripts/coco_dog_qwen3vl4b_zimage.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/coco_dog_qwen3vl4b_zimage.sh   # only GPU 2
#   experiment_scripts/coco_dog_qwen3vl4b_zimage.sh --status | --stop

set -euo pipefail
export DATASET="coco_dog"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_qwen3vl4b_zimage.sh" "$@"
