#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — reproduces the results reported in the paper.
#
# Run from the repository root:  bash run_pipeline.sh
#
# Steps 1-4 are quick. Step 5 trains and takes roughly 2.5-3 hours per seed
# that does not already have a model directory.
# =============================================================================
set -euo pipefail

mkdir -p logs data results

echo "=== 1/5  Recover IndicQA answer spans ==============================="
# The stored answer_start offsets are unreliable. This locates each answer by
# substring search instead, recovering 1,392 of 1,398 answerable examples.
python3 fixes/indicqa_offset_fix.py \
    --in  data/indicqa_raw.te.json \
    --out data/indicqa_telugu_clean.json | tee logs/01_recover.log

echo
echo "=== 2/5  Merge the two sources ======================================"
# Produces data/telugu_math_{train,val,test}.json from the recovered IndicQA
# and the translated MAWPS set.
#
# NOTE: check the flags below against your own 04_merge_data.py before relying
# on this line — it is the one step whose interface is not pinned down here.
python3 data/04_merge_data.py | tee logs/02_merge.log

echo
echo "=== 3/5  Re-split by context group =================================="
# A naive example-level shuffle leaks near-duplicate MAWPS contexts across the
# split. This shuffles whole context groups instead, 80/10/10, seed 42.
python3 fixes/regroup_split.py \
    --train  data/telugu_math_train.json \
    --val    data/telugu_math_val.json \
    --test   data/telugu_math_test.json \
    --outdir data \
    --seed   42 | tee logs/03_regroup.log

echo
echo "=== 4/5  Audit the split for leakage ================================"
python3 fixes/leakage_audit.py \
    --train data/telugu_math_train_grouped.json \
    --test  data/telugu_math_test_grouped.json \
    --val   data/telugu_math_val_grouped.json | tee logs/04_leakage.log

echo
echo ">>> Read logs/04_leakage.log before going further. If near-duplicate"
echo ">>> overlap between train and test is high, the split is not clean and"
echo ">>> the numbers from step 5 will be inflated."
echo

echo "=== 5/5  Train and evaluate across seeds ============================"
# Seeds 42, 123 and 7. A seed whose model directory already exists is only
# re-evaluated, not retrained, so the original seed-42 run can be reused:
#     cp -r model/muril_telugu_math_v2 model/muril_seed42
#
# For a few-minute sanity check over all three seeds first:
#     SMOKE=1 python3 fixes/run_multiseed.py
python3 fixes/run_multiseed.py | tee logs/05_multiseed.log

echo
echo "Done. Results are in results/multiseed_results.json"
echo "Paper reports: IndicQA 57.7 F1 / 47.3 EM (seed 42), mean 59.1 +/- 2.5"
echo "               MAWPS 100.0 F1, matched by the rule-based baseline"
echo "               rule-based baseline on IndicQA: 2.7 F1"
