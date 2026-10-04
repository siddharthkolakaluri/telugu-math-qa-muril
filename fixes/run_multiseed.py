"""
run_multiseed.py
----------------
Train the corrected windowed model with SEVERAL random seeds and evaluate each
on the leakage-free grouped test set, then report mean +/- std for the metrics
that matter. This turns the single-seed result into a variance-aware one, which
is the single biggest weakness the paper currently discloses.

It reuses the EXACT training and evaluation logic from finetune_windowed.py and
evaluate_windowed.py, so the numbers are directly comparable to what you already
have. Seed 42 is included, so your existing result is one of the three runs.

Design:
  * SEEDS = [42, 123, 7]. Each trains into its own dir  model/muril_seed{N}.
  * If a seed's model dir already exists, training is SKIPPED and it is only
    re-evaluated. So you can reuse your existing seed-42 model:
        cp -r model/muril_telugu_math_v2 model/muril_seed42
    (do this first to save one ~2.5h training run).
  * Each run is wrapped in try/except, so one failure won't lose the others.
  * Results are written to  results/multiseed_results.json  and printed as a
    per-seed table plus a mean +/- std summary you can paste back.

Quick sanity check first (a ~few-minute tiny run over all seeds):
    SMOKE=1 python3 fixes/run_multiseed.py

The real run (leave it going; ~2.5-3h per seed that needs training):
    python3 fixes/run_multiseed.py
"""
import json, os, re, string, collections, time, statistics, traceback
import numpy as np, torch
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, EarlyStoppingCallback,
                          default_data_collator, set_seed)
from torch.utils.data import Dataset

# ------------------------------------------------------------------ config
ROOT        = os.path.expanduser("~/Documents/nlp_project")
MODEL_NAME  = "google/muril-base-cased"
SEEDS       = [42, 123, 7]
MAX_LENGTH, DOC_STRIDE = 384, 128
LR, WEIGHT_DECAY, BATCH = 2e-5, 0.01, 4
MAX_EPOCHS  = 30                       # early stopping ends it sooner
SMOKE       = os.environ.get("SMOKE") == "1"

TRAIN_PATH = os.path.join(ROOT, "data", "telugu_math_train_grouped.json")
VAL_PATH   = os.path.join(ROOT, "data", "telugu_math_val_grouped.json")
TEST_PATH  = os.path.join(ROOT, "data", "telugu_math_test_grouped.json")
RESULTS_DIR = os.path.join(ROOT, "results")
N_BEST, MAX_ANS_TOK = 20, 30
ANSWER_CUES = ["సమాధానం", "The answer is"]
MAWPS_Q = ["సమాధానం ఏమిటి", "what is the answer"]


def device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ------------------------------------------------------------ training dataset
# (identical logic to finetune_windowed.py: keep the one answer-bearing window)
class QADataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.tok = tokenizer
        self.features = []
        self.with_answer = self.skipped = 0
        for ex in examples:
            f = self._one(ex)
            if f is not None: self.features.append(f)
            else: self.skipped += 1
        print(f"   features: {len(self.features):,}  "
              f"(answer-bearing {self.with_answer}, skipped {self.skipped})")

    def _one(self, ex):
        q, ctx, ans = ex["question"], ex["context"], ex["answer_text"]
        cstart = ex["answer_start"]; cend = cstart + len(ans)
        enc = self.tok(q, ctx, max_length=MAX_LENGTH, truncation="only_second",
                       stride=DOC_STRIDE, return_overflowing_tokens=True,
                       return_offsets_mapping=True, padding="max_length")
        for w in range(len(enc["input_ids"])):
            seq = enc.sequence_ids(w)
            offs = enc["offset_mapping"][w]
            ids = enc["input_ids"][w]
            t0 = 0
            while t0 < len(seq) and seq[t0] != 1: t0 += 1
            t1 = len(seq) - 1
            while t1 >= 0 and seq[t1] != 1: t1 -= 1
            if t0 >= len(seq) or t1 < 0: continue
            if not (offs[t0][0] <= cstart and offs[t1][1] >= cend): continue
            s = t0
            while s <= t1 and offs[s][0] <= cstart: s += 1
            start_pos = s - 1
            e = t1
            while e >= t0 and offs[e][1] >= cend: e -= 1
            end_pos = e + 1
            if start_pos < t0 or end_pos > t1 or start_pos > end_pos: continue
            self.with_answer += 1
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"][w], dtype=torch.long),
                "token_type_ids": torch.tensor(enc["token_type_ids"][w], dtype=torch.long),
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long),
            }
        return None

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]


def load(path, label):
    data = json.load(open(path, encoding="utf-8"))
    print(f"loaded {label}: {len(data):,}")
    return data


def train_one(seed, out_dir, tok, train_ds, val_ds, dev):
    set_seed(seed)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(dev)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1 if SMOKE else MAX_EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        logging_steps=25, report_to="none", fp16=(dev == "cuda"),
        dataloader_pin_memory=False, seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=default_data_collator,
                      callbacks=[] if SMOKE else [EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time()
    trainer.train()
    trainer.save_model(out_dir); tok.save_pretrained(out_dir)
    mins = (time.time() - t0) / 60
    evals = [(h["epoch"], h["eval_loss"]) for h in trainer.state.log_history if "eval_loss" in h]
    best_epoch, best_val = min(evals, key=lambda x: x[1]) if evals else (None, None)
    print(f"   seed {seed}: {mins:.1f} min, best val loss {best_val} at epoch ~{best_epoch}, "
          f"{len(evals)} epochs ran")
    return {"minutes": round(mins, 1), "epochs_ran": len(evals),
            "best_epoch": best_epoch, "best_val_loss": best_val}


# --------------------------------------------------------------- eval metrics
def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())

def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0 or not p or not g: return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)

def em_score(pred, gold):
    return float(normalize(pred) == normalize(gold))

def get_gold(ex):
    if "answers" in ex and ex["answers"]:
        a = ex["answers"]
        return (a[0]["text"] if isinstance(a, list) else a["text"])
    return ex.get("answer_text", ex.get("answer", ""))

def get_source(ex):
    if ex.get("source"):
        return "mawps" if "mawps" in ex["source"].lower() else "indicqa"
    q = (ex.get("question") or "").lower()
    return "mawps" if any(m in q for m in MAWPS_Q) else "indicqa"

def rule_based(context):
    for cue in ANSWER_CUES:
        i = context.rfind(cue)
        if i != -1:
            m = re.search(r"-?\d+(?:[.,]\d+)?", context[i + len(cue):])
            if m: return m.group(0)
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", context)
    return nums[-1] if nums else ""


def build_predictor(model_dir, dev):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).eval().to(dev)
    def predict(question, context):
        enc = tok(question, context, max_length=MAX_LENGTH,
                  truncation="only_second", stride=DOC_STRIDE,
                  return_overflowing_tokens=True, return_offsets_mapping=True,
                  padding="max_length", return_tensors="pt")
        n = enc["input_ids"].shape[0]
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev),
                        token_type_ids=enc.get("token_type_ids").to(dev)
                        if "token_type_ids" in enc else None)
        starts, ends = out.start_logits.cpu().numpy(), out.end_logits.cpu().numpy()
        best_text, best_score = "", -1e30
        for w in range(n):
            seq = enc.sequence_ids(w)
            offs = enc["offset_mapping"][w].tolist()
            s_logit, e_logit = starts[w], ends[w]
            ctx_idx = [i for i, sid in enumerate(seq) if sid == 1]
            if not ctx_idx: continue
            for si in np.argsort(s_logit)[-N_BEST:][::-1]:
                if si not in ctx_idx: continue
                for ei in np.argsort(e_logit)[-N_BEST:][::-1]:
                    if ei not in ctx_idx or ei < si or ei - si + 1 > MAX_ANS_TOK: continue
                    score = s_logit[si] + e_logit[ei]
                    if score > best_score:
                        best_score = score
                        best_text = context[offs[si][0]:offs[ei][1]]
                    break
        return best_text.strip()
    return predict


def score(examples, predict_fn):
    buckets = collections.defaultdict(lambda: {"f1": 0.0, "em": 0.0, "n": 0})
    for ex in examples:
        gold, pred, src = get_gold(ex), predict_fn(ex), get_source(ex)
        for key in (src, "all"):
            buckets[key]["f1"] += f1_score(pred, gold)
            buckets[key]["em"] += em_score(pred, gold)
            buckets[key]["n"] += 1
    return {k: {"n": v["n"],
                "f1": round(100 * v["f1"] / max(v["n"], 1), 1),
                "em": round(100 * v["em"] / max(v["n"], 1), 1)}
            for k, v in buckets.items()}


# --------------------------------------------------------------------- driver
def main():
    dev = device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 64)
    print(f"device: {dev}   SMOKE={SMOKE}   seeds: {SEEDS}")
    print("=" * 64)

    train = load(TRAIN_PATH, "train"); val = load(VAL_PATH, "val")
    test = json.load(open(TEST_PATH, encoding="utf-8"))
    print(f"loaded test: {len(test):,}")
    if SMOKE:
        train, val, test = train[:200], val[:60], test[:60]
        print(">> SMOKE MODE: tiny slice, 1 epoch per seed")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("\ntokenizing train..."); train_ds = QADataset(train, tok)
    print("tokenizing val...");    val_ds = QADataset(val, tok)

    # rule-based baseline: same for every seed, compute once
    baseline = score(test, lambda ex: rule_based(ex["context"]))
    print(f"\nrule-based baseline -> MAWPS F1 {baseline.get('mawps',{}).get('f1')}, "
          f"IndicQA F1 {baseline.get('indicqa',{}).get('f1')}")

    runs = []
    for seed in SEEDS:
        out_dir = os.path.join(ROOT, "model", f"muril_seed{seed}")
        print("\n" + "-" * 64)
        print(f"SEED {seed}  ->  {out_dir}")
        rec = {"seed": seed, "model_dir": out_dir}
        try:
            if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "config.json")):
                print("   model dir exists -> skipping training, evaluating only")
                rec["trained"] = False
            else:
                rec.update(train_one(seed, out_dir, tok, train_ds, val_ds, dev))
                rec["trained"] = True
            predict = build_predictor(out_dir, dev)
            m = score(test, lambda ex: predict(ex["question"], ex["context"]))
            rec["metrics"] = m
            print(f"   -> IndicQA F1 {m['indicqa']['f1']} EM {m['indicqa']['em']} | "
                  f"MAWPS F1 {m['mawps']['f1']} | all F1 {m['all']['f1']}")
        except Exception as ex:
            rec["error"] = repr(ex)
            print("   !! seed failed:", ex)
            traceback.print_exc()
        runs.append(rec)
        json.dump({"baseline": baseline, "runs": runs},
                  open(os.path.join(RESULTS_DIR, "multiseed_results.json"), "w"),
                  ensure_ascii=False, indent=2)

    # ---- summary ----
    ok = [r for r in runs if "metrics" in r]
    def col(metric, subset):
        return [r["metrics"][subset][metric] for r in ok]

    print("\n" + "=" * 64)
    print("PER-SEED RESULTS")
    print("=" * 64)
    print(f"{'seed':>6} | {'IndicQA F1':>11} {'IndicQA EM':>11} | {'MAWPS F1':>9} | {'all F1':>7}")
    for r in ok:
        m = r["metrics"]
        print(f"{r['seed']:>6} | {m['indicqa']['f1']:>11} {m['indicqa']['em']:>11} | "
              f"{m['mawps']['f1']:>9} | {m['all']['f1']:>7}")

    def ms(vals):
        if not vals: return "n/a"
        if len(vals) == 1: return f"{vals[0]:.1f} (1 run)"
        return f"{statistics.mean(vals):.1f} +/- {statistics.stdev(vals):.1f}"

    print("\n" + "=" * 64)
    print("MEAN +/- STD  (n =", len(ok), "seeds)")
    print("=" * 64)
    print(f"  Rule-based baseline  IndicQA F1 {baseline['indicqa']['f1']}   MAWPS F1 {baseline['mawps']['f1']}")
    print(f"  Fine-tuned  IndicQA  F1 {ms(col('f1','indicqa'))}   EM {ms(col('em','indicqa'))}")
    print(f"  Fine-tuned  MAWPS    F1 {ms(col('f1','mawps'))}   EM {ms(col('em','mawps'))}")
    print(f"  Fine-tuned  overall  F1 {ms(col('f1','all'))}   EM {ms(col('em','all'))}")
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'multiseed_results.json')}")
    print("Paste this whole summary back and I'll fold mean +/- std into the paper.")


if __name__ == "__main__":
    main()
