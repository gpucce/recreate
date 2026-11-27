#!/usr/bin/env bash
# Pair classifier (direct copy vs multiple reuses) with default settings on every available generation run,
# one run at a time on GPU 2. Logs in logs/, results in outputs/pair_classifier/<dataset>/<generator>[_alternate]/.
#
#   bash experiment_scripts/pair_classifier_all.sh

D=/raid/datasets/content_convergence/da_mettere_in_homeprogetto/outputs/textonly-generations
O=outputs/pair_classifier
cd "$(dirname "$0")/.."
mkdir -p logs

CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/bbc-news/gemma-4-31b > logs/pair_classifier_bbc-news_gemma-4-31b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/bbc-news/ministral-3-14b > logs/pair_classifier_bbc-news_ministral-3-14b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/bbc-news/qwen_qwen3.8_27b/bbc-news-with_generations.csv --output-dir $O/bbc-news/qwen3.8-27b > logs/pair_classifier_bbc-news_qwen3.8-27b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/erc-grants/gemma-4-31b > logs/pair_classifier_erc-grants_gemma-4-31b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/erc-grants/qwen_qwen3.8_27b/erc_grants.with_generations.csv --output-dir $O/erc-grants/qwen3.8-27b > logs/pair_classifier_erc-grants_qwen3.8-27b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/erc-grants/ministral-3-14b > logs/pair_classifier_erc-grants_ministral-3-14b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/acl/gemma-4-31b > logs/pair_classifier_acl_gemma-4-31b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/acl/ministral-3-14b > logs/pair_classifier_acl_ministral-3-14b.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/acl/qwen3.8-27b > logs/pair_classifier_acl_qwen3.8-27b.log 2>&1

# Odd steps (positions 1, 3, 5, ...: ERC full proposals / ACL reviews / BBC alternate texts)
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/bbc-news/google_gemma-4-31b-it/bbc-news.with_generations.csv --output-dir $O/bbc-news/gemma-4-31b_alternate > logs/pair_classifier_bbc-news_gemma-4-31b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/bbc-news/mistralai_ministral-3-14b-instruct-2512/bbc-news.with_generations.csv --output-dir $O/bbc-news/ministral-3-14b_alternate > logs/pair_classifier_bbc-news_ministral-3-14b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/bbc-news/qwen_qwen3.8_27b/bbc-news-with_generations.csv --output-dir $O/bbc-news/qwen3.8-27b_alternate > logs/pair_classifier_bbc-news_qwen3.8-27b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/erc-grants/google_gemma-4-31b-it/erc_grants.with_generations.csv --output-dir $O/erc-grants/gemma-4-31b_alternate > logs/pair_classifier_erc-grants_gemma-4-31b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/erc-grants/qwen_qwen3.8_27b/erc_grants.with_generations.csv --output-dir $O/erc-grants/qwen3.8-27b_alternate > logs/pair_classifier_erc-grants_qwen3.8-27b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/erc-grants/mistralai_ministral-3-14b-instruct-2512/erc_grants.with_generations.csv --output-dir $O/erc-grants/ministral-3-14b_alternate > logs/pair_classifier_erc-grants_ministral-3-14b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/acl/google_gemma-4-31b-it/acl.with_generations.csv --output-dir $O/acl/gemma-4-31b_alternate > logs/pair_classifier_acl_gemma-4-31b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/acl/mistralai_ministral-3-14b-instruct-2512/acl.with_generations.csv --output-dir $O/acl/ministral-3-14b_alternate > logs/pair_classifier_acl_ministral-3-14b_alternate.log 2>&1
CUDA_VISIBLE_DEVICES=2 nohup python pair_classifier.py --text-type alternate --data $D/acl/qwen_qwen3.8-27b/acl.with_generations.csv --output-dir $O/acl/qwen3.8-27b_alternate > logs/pair_classifier_acl_qwen3.8-27b_alternate.log 2>&1
