"""
finetune_windowed.py
--------------------
Corrected fine-tuning for the rebalanced, leakage-free dataset.

Fixes vs the original 02_finetune.py:
  * searches ALL sliding windows for the answer (original kept only window 0,
    silently dropping long-context IndicQA examples)
  * keeps the single answer-bearing window per example -> training stays fast
  * EarlyStoppingCallback(patience=3) so it stops ~3 epochs after val loss
    stops improving instead of grinding all the way to 100
  * SMOKE test switch for a 2-minute end-to-end sanity run

Reads the GROUPED (leakage-free) splits by default.
Writes to model/muril_telugu_math_v2 so your existing model is untouched.

Run a smoke test first:   SMOKE=1 python3 fixes/finetune_windowed.py
Then the real run:              python3 fixes/finetune_windowed.py
"""
import json, os, time, torch
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, EarlyStoppingCallback,
                          default_data_collator, set_seed)
from torch.utils.data import Dataset

ROOT = os.path.expanduser("~/Documents/nlp_project")
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH, DOC_STRIDE = 384, 128
LR, WEIGHT_DECAY, BATCH = 2e-5, 0.01, 4
MAX_EPOCHS = 30                      # early stopping will end it sooner
SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"

TRAIN_PATH = os.path.join(ROOT, "data", "telugu_math_train_grouped.json")
VAL_PATH   = os.path.join(ROOT, "data", "telugu_math_val_grouped.json")
OUT_DIR    = os.path.join(ROOT, "model", "_smoke" if SMOKE else "muril_telugu_math_v2")


def device():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


class QADataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.tok = tokenizer
        self.features = []
        self.with_answer = self.cls_fallback = self.skipped = 0
        for ex in examples:
            f = self._one(ex)
            if f is not None:
                self.features.append(f)
            else:
                self.skipped += 1
        print(f"   features: {len(self.features):,}  "
              f"(answer-bearing {self.with_answer}, skipped-no-window {self.skipped})")

    def _one(self, ex):
        q, ctx, ans = ex["question"], ex["context"], ex["answer_text"]
        cstart = ex["answer_start"]; cend = cstart + len(ans)
        enc = self.tok(q, ctx, max_length=MAX_LENGTH, truncation="only_second",
                       stride=DOC_STRIDE, return_overflowing_tokens=True,
                       return_offsets_mapping=True, padding="max_length")
        n_windows = len(enc["input_ids"])
        for w in range(n_windows):
            seq = enc.sequence_ids(w)
            offs = enc["offset_mapping"][w]
            ids = enc["input_ids"][w]
            cls_index = ids.index(self.tok.cls_token_id) if self.tok.cls_token_id in ids else 0
            # context token span within this window
            t0 = 0
            while t0 < len(seq) and seq[t0] != 1: t0 += 1
            t1 = len(seq) - 1
            while t1 >= 0 and seq[t1] != 1: t1 -= 1
            if t0 >= len(seq) or t1 < 0:
                continue
            # is the whole answer inside this window's context?
            if not (offs[t0][0] <= cstart and offs[t1][1] >= cend):
                continue
            s = t0
            while s <= t1 and offs[s][0] <= cstart: s += 1
            start_pos = s - 1
            e = t1
            while e >= t0 and offs[e][1] >= cend: e -= 1
            end_pos = e + 1
            if start_pos < t0 or end_pos > t1 or start_pos > end_pos:
                continue
            self.with_answer += 1
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"][w], dtype=torch.long),
                "token_type_ids": torch.tensor(enc["token_type_ids"][w], dtype=torch.long),
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long),
            }
        return None  # answer not found in any window

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]


def load(path, label):
    data = json.load(open(path, encoding="utf-8"))
    print(f"loaded {label}: {len(data):,}  ({path})")
    return data


def main():
    set_seed(SEED)
    dev = device()
    print("=" * 60)
    print(f"device: {dev}   SMOKE={SMOKE}   output: {OUT_DIR}")
    print("=" * 60)

    train = load(TRAIN_PATH, "train")
    val = load(VAL_PATH, "val")
    if SMOKE:
        train, val = train[:200], val[:60]
        print(">> SMOKE MODE: using a tiny slice for 1 epoch")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("\ntokenizing train..."); train_ds = QADataset(train, tok)
    print("tokenizing val...");     val_ds = QADataset(val, tok)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(dev)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1 if SMOKE else MAX_EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        learning_rate=LR, weight_decay=WEIGHT_DECAY, warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=1,
        logging_steps=25, report_to="none", fp16=(dev == "cuda"),
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=default_data_collator,
                      callbacks=[] if SMOKE else [EarlyStoppingCallback(early_stopping_patience=3)])

    t0 = time.time()
    trainer.train()
    trainer.save_model(OUT_DIR); tok.save_pretrained(OUT_DIR)
    mins = (time.time() - t0) / 60
    print(f"\nDONE in {mins:.1f} min. best val loss: "
          f"{min([h['eval_loss'] for h in trainer.state.log_history if 'eval_loss' in h], default='?')}")
    print(f"model saved to {OUT_DIR}")
    if SMOKE:
        print("\nSmoke test passed if you see decreasing loss and a saved model.")
        print("Now run the REAL training:  python3 fixes/finetune_windowed.py")


if __name__ == "__main__":
    main()
