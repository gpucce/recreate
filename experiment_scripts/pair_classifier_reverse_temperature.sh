#!/usr/bin/env bash
# Pair classifier with --reverse (4 classes) on the temperature generations, spread over GPUs 4-7:
# one queue per GPU, run in parallel; inside a queue the runs go one after another, so every run uses a single GPU.
# Each GPU gets one generator for every dataset and step type (6 runs per GPU).
# Same setup as pair_classifier_reverse_all.sh. Logs in logs/temperature/,
# results in outputs/pair_classifier_temperature/<dataset>/<generator>_reverse[_alternate]/.
#
#   nohup bash experiment_scripts/pair_classifier_reverse_temperature.sh > logs/temperature_all.log 2>&1 &

D=data_bkp/da_mettere_in_homeprogetto/outputs/textonly_generations_temperature
O=outputs/pair_classifier_temperature
cd "$(dirname "$0")/.."
mkdir -p logs/temperature

# GPU 4
(
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/bbc-news/gemma-4-31b_reverse --reverse > logs/temperature/pair_classifier_bbc-news_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/erc-grants/gemma-4-31b_reverse --reverse > logs/temperature/pair_classifier_erc-grants_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/acl/gemma-4-31b_reverse --reverse > logs/temperature/pair_classifier_acl_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --text-type alternate --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/bbc-news/gemma-4-31b_reverse_alternate --reverse > logs/temperature/pair_classifier_bbc-news_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --text-type alternate --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/erc-grants/gemma-4-31b_reverse_alternate --reverse > logs/temperature/pair_classifier_erc-grants_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=4 python pair_classifier.py --text-type alternate --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/acl/gemma-4-31b_reverse_alternate --reverse > logs/temperature/pair_classifier_acl_gemma-4-31b_reverse_alternate.log 2>&1
) &

# GPU 5
(
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/bbc-news/ministral-3-14b_reverse --reverse > logs/temperature/pair_classifier_bbc-news_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/erc-grants/ministral-3-14b_reverse --reverse > logs/temperature/pair_classifier_erc-grants_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/acl/ministral-3-14b_reverse --reverse > logs/temperature/pair_classifier_acl_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --text-type alternate --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/bbc-news/ministral-3-14b_reverse_alternate --reverse > logs/temperature/pair_classifier_bbc-news_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --text-type alternate --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/erc-grants/ministral-3-14b_reverse_alternate --reverse > logs/temperature/pair_classifier_erc-grants_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=5 python pair_classifier.py --text-type alternate --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/acl/ministral-3-14b_reverse_alternate --reverse > logs/temperature/pair_classifier_acl_ministral-3-14b_reverse_alternate.log 2>&1
) &

# GPU 6
(
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/bbc-news/mistral-small-3.2-24b_reverse --reverse > logs/temperature/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/erc-grants/mistral-small-3.2-24b_reverse --reverse > logs/temperature/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/acl/mistral-small-3.2-24b_reverse --reverse > logs/temperature/pair_classifier_acl_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --text-type alternate --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/bbc-news/mistral-small-3.2-24b_reverse_alternate --reverse > logs/temperature/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --text-type alternate --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/erc-grants/mistral-small-3.2-24b_reverse_alternate --reverse > logs/temperature/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=6 python pair_classifier.py --text-type alternate --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/acl/mistral-small-3.2-24b_reverse_alternate --reverse > logs/temperature/pair_classifier_acl_mistral-small-3.2-24b_reverse_alternate.log 2>&1
) &

# GPU 7
(
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/bbc-news/qwen3.8-27b_reverse --reverse > logs/temperature/pair_classifier_bbc-news_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/erc-grants/qwen3.8-27b_reverse --reverse > logs/temperature/pair_classifier_erc-grants_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/acl/qwen3.8-27b_reverse --reverse > logs/temperature/pair_classifier_acl_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --text-type alternate --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/bbc-news/qwen3.8-27b_reverse_alternate --reverse > logs/temperature/pair_classifier_bbc-news_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --text-type alternate --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/erc-grants/qwen3.8-27b_reverse_alternate --reverse > logs/temperature/pair_classifier_erc-grants_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=7 python pair_classifier.py --text-type alternate --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/acl/qwen3.8-27b_reverse_alternate --reverse > logs/temperature/pair_classifier_acl_qwen3.8-27b_reverse_alternate.log 2>&1
) &

wait
