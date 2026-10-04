"""
Emergency merge script — run this directly in a notebook cell.
Combines IndicQA + MAWPS and saves train/val/test splits.
"""

import json, os, random

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "/Users/siddharth/Documents/nlp_project/data"
INDICQA    = os.path.join(DATA_DIR, "indicqa_telugu_clean.json")
MAWPS      = os.path.join(DATA_DIR, "mawps_telugu_clean.json")
TRAIN_OUT  = os.path.join(DATA_DIR, "telugu_math_train.json")
VAL_OUT    = os.path.join(DATA_DIR, "telugu_math_val.json")
TEST_OUT   = os.path.join(DATA_DIR, "telugu_math_test.json")

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10

# ── Load ───────────────────────────────────────────────────────────────────
with open(INDICQA, encoding="utf-8") as f:
    indicqa = json.load(f)
with open(MAWPS, encoding="utf-8") as f:
    mawps = json.load(f)

print(f"Loaded  IndicQA : {len(indicqa):,}")
print(f"Loaded  MAWPS   : {len(mawps):,}")

# ── Validate ───────────────────────────────────────────────────────────────
# For IndicQA: strict span check (these were verified at download time)
# For MAWPS  : light check only — translated contexts may have minor
#              encoding differences so we just verify the answer number
#              appears somewhere in the context rather than exact position.

def validate_example(ex):
    """Return True if the example is usable for training."""
    required = ["id", "context", "question", "answer_text", "answer_start"]
    if not all(k in ex for k in required):
        return False
    if not ex["context"].strip() or not ex["question"].strip():
        return False
    if not ex["answer_text"].strip():
        return False

    start = ex["answer_start"]
    end   = start + len(ex["answer_text"])
    ctx   = ex["context"]

    # answer_start must be a plausible position
    if start < 0 or end > len(ctx):
        return False

    source = ex.get("source", "")

    if source == "indicqa":
        # Strict: the exact span must match
        return ctx[start:end] == ex["answer_text"]

    elif source == "mawps_translated":
        # Relaxed: the answer (a number) must appear SOMEWHERE in the context.
        # Exact position can drift slightly after Google Translate.
        # We also re-anchor answer_start to the correct position here.
        answer = ex["answer_text"]
        pos    = ctx.rfind(answer)   # find last occurrence (appended at end)
        if pos < 0:
            return False
        # Fix the answer_start to the verified position
        ex["answer_start"] = pos
        return True

    else:
        # Unknown source — apply strict check
        return ctx[start:end] == ex["answer_text"]


valid   = []
removed = 0

for ex in indicqa + mawps:
    if validate_example(ex):
        valid.append(ex)
    else:
        removed += 1

print(f"\nValidation:")
print(f"  Passed  : {len(valid):,}")
print(f"  Removed : {removed:,}")

# ── Deduplicate ────────────────────────────────────────────────────────────
# IndicQA  → deduplicate on question (each question is unique)
# MAWPS    → deduplicate on context  (all questions are identical:
#            "సమాధానం ఏమిటి?" so question-based dedup kills 299/300 rows)

seen   = set()
unique = []
for ex in valid:
    if ex.get("source") == "mawps_translated":
        key = ex["context"].strip()[:100]   # use first 100 chars of context
    else:
        key = ex["question"].lower().strip() # use question for IndicQA
    if key not in seen:
        seen.add(key)
        unique.append(ex)

print(f"  After dedup: {len(unique):,}")

# ── Shuffle + Split ────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
random.shuffle(unique)

total     = len(unique)
train_end = int(total * TRAIN_RATIO)
val_end   = train_end + int(total * VAL_RATIO)

train = unique[:train_end]
val   = unique[train_end:val_end]
test  = unique[val_end:]

print(f"\nSplit (seed={RANDOM_SEED}):")
print(f"  Train      : {len(train):,}  (80%)")
print(f"  Validation : {len(val):,}  (10%)")
print(f"  Test       : {len(test):,}  (10%)")

# Source breakdown
for src in ["indicqa", "mawps_translated"]:
    t = sum(1 for e in train if e.get("source") == src)
    v = sum(1 for e in val   if e.get("source") == src)
    s = sum(1 for e in test  if e.get("source") == src)
    print(f"  {src:20s}: {t} / {v} / {s}  (train/val/test)")

# ── Save ───────────────────────────────────────────────────────────────────
for path, data, label in [
    (TRAIN_OUT, train, "train"),
    (VAL_OUT,   val,   "val"),
    (TEST_OUT,  test,  "test"),
]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size = os.path.getsize(path)
    print(f"  ✅ Saved {label:6s}: {len(data):,} examples → {path} ({size:,} bytes)")

# ── Quick verify ───────────────────────────────────────────────────────────
with open(TRAIN_OUT, encoding="utf-8") as f:
    check = json.load(f)
print(f"\nVerification: reloaded {len(check):,} training examples ✅")
print(f"Sample question : {check[0]['question']}")
print(f"Sample answer   : {check[0]['answer_text']}")
print(f"\n✅ DONE — data is ready for fine-tuning!")
