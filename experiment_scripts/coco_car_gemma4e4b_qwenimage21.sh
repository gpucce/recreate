#!/usr/bin/env bash
#
# coco_car (1000 COCO objects, images in data_bkp/coco/) with the gemma4e4b_qwenimage21 models: same experiment,
# GPU layout and env vars as vlstereoset_gemma4e4b_qwenimage21.sh, only the dataset changes. Output goes to
# data_bkp/coco_car_experiments/gemma4e4b_qwenimage21/<run name>/.
#
#   experiment_scripts/coco_car_gemma4e4b_qwenimage21.sh --dry-run    # plan only
#   experiment_scripts/coco_car_gemma4e4b_qwenimage21.sh              # launch in the background
#   GPU_GROUPS="2" experiment_scripts/coco_car_gemma4e4b_qwenimage21.sh   # only GPU 2
#   experiment_scripts/coco_car_gemma4e4b_qwenimage21.sh --status | --stop

set -euo pipefail
export DATASET="coco_car"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vlstereoset_gemma4e4b_qwenimage21.sh" "$@"
