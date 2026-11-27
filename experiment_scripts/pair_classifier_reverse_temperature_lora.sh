#!/usr/bin/env bash
# Pair classifier with --reverse (4 classes) on the temperature generations, with two LoRA-tuned decoder models
# as the classifier instead of ModernBERT:
#   Qwen/Qwen3-VL-Embedding-2B -> outputs/pair_classifier_temperature_lora/qwen3-vl-embedding-2b/<dataset>/<generator>_reverse[_alternate]/
#   google/gemma-4-E2B         -> outputs/pair_classifier_temperature_lora/gemma-4-e2b/<dataset>/<generator>_reverse[_alternate]/
# Logs in logs/temperature_lora/<model>/. Same pairs and splits as pair_classifier_reverse_temperature.sh;
# batch 8 (as ModernBERT), lr 1e-4, LoRA r=16, 3 epochs, 8192 tokens per pair.
#
# 48 runs (2 models x 4 generators x 3 datasets x 2 step types) on GPUs 2-7, one queue per GPU run in parallel;
# inside a queue the runs go one after another, so every run uses a single GPU. Runs were assigned to the queues
# longest first, from the smoke-test estimates (hours per run, on the original generations):
#   Qwen  bbc 4.2/2.6, erc 1.1/11.2, acl 8.8/8.7 (even/odd steps)  -> 36.6 h per generator (peak 27 GB)
#   Gemma bbc 7.7/4.3, erc 1.8/26.3, acl 18.1/18.0                 -> 76.1 h per generator (peak 44 GB)
# Total ~451 GPU-hours; the longest queue is ~76 h (~3.1 days).
#
#   nohup bash experiment_scripts/pair_classifier_reverse_temperature_lora.sh > logs/temperature_lora_all.log 2>&1 &
# Stop everything (script, queues and runs) with: kill -- -<PID of the script>

D=data_bkp/da_mettere_in_homeprogetto/outputs/textonly_generations_temperature
O=outputs/pair_classifier_temperature_lora
L=logs/temperature_lora
QWEN="--lora --model Qwen/Qwen3-VL-Embedding-2B --batch-size 8"
GEMMA="--lora --model google/gemma-4-E2B --batch-size 8"
cd "$(dirname "$0")/.."
mkdir -p $L/qwen3-vl-embedding-2b $L/gemma-4-e2b

# GPU 2: ~76 h
(
GPU=2
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/gemma-4-31b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/gemma-4-31b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_acl_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/mistral-small-3.2-24b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/mistral-small-3.2-24b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/mistral-small-3.2-24b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/mistral-small-3.2-24b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/mistral-small-3.2-24b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/mistral-small-3.2-24b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse.log 2>&1
) &

# GPU 3: ~76 h
(
GPU=3
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/ministral-3-14b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/ministral-3-14b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_acl_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/qwen3.8-27b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/qwen3.8-27b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/qwen3.8-27b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/qwen3.8-27b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/qwen3.8-27b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/qwen3.8-27b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_qwen3.8-27b_reverse.log 2>&1
) &

# GPU 4: ~75 h
(
GPU=4
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/erc-grants/mistralai_mistral-small-3.2-24b-instruct-2506/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/mistral-small-3.2-24b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/mistral-small-3.2-24b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_acl_mistral-small-3.2-24b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/gemma-4-31b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/gemma-4-31b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/mistral-small-3.2-24b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/gemma-4-31b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/bbc-news/mistralai_mistral-small-3.2-24b-instruct-2506/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/mistral-small-3.2-24b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_mistral-small-3.2-24b_reverse_alternate.log 2>&1
) &

# GPU 5: ~75 h
(
GPU=5
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/erc-grants/qwen_qwen3.8-27b/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/qwen3.8-27b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/qwen3.8-27b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_acl_qwen3.8-27b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/ministral-3-14b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/ministral-3-14b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/qwen3.8-27b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/ministral-3-14b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/bbc-news/qwen_qwen3.8-27b/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/qwen3.8-27b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_qwen3.8-27b_reverse_alternate.log 2>&1
) &

# GPU 6: ~75 h
(
GPU=6
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/gemma-4-31b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_acl_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/mistral-small-3.2-24b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_acl_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/gemma-4-31b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/acl/mistralai_mistral-small-3.2-24b-instruct-2506/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/mistral-small-3.2-24b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_mistral-small-3.2-24b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/gemma-4-31b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/gemma-4-31b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_gemma-4-31b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/gemma-4-31b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/gemma-4-31b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_gemma-4-31b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/gemma-4-31b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_gemma-4-31b_reverse.log 2>&1
) &

# GPU 7: ~75 h
(
GPU=7
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/ministral-3-14b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_acl_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/gemma-4-e2b/acl/qwen3.8-27b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_acl_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --text-type alternate --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/ministral-3-14b_reverse_alternate --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/acl/qwen3.8-27b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_acl_qwen3.8-27b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/ministral-3-14b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --text-type alternate --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/gemma-4-e2b/bbc-news/ministral-3-14b_reverse_alternate --reverse > $L/gemma-4-e2b/pair_classifier_bbc-news_ministral-3-14b_reverse_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/bbc-news/ministral-3-14b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_bbc-news_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $GEMMA --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/gemma-4-e2b/erc-grants/ministral-3-14b_reverse --reverse > $L/gemma-4-e2b/pair_classifier_erc-grants_ministral-3-14b_reverse.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU python pair_classifier.py $QWEN --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/qwen3-vl-embedding-2b/erc-grants/ministral-3-14b_reverse --reverse > $L/qwen3-vl-embedding-2b/pair_classifier_erc-grants_ministral-3-14b_reverse.log 2>&1
) &

wait
