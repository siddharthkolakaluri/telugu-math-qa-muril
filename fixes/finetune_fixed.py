"""
finetune_fixed.py
-----------------
Drop-in replacement for 03_model/02_finetune.py that fixes the training-side
limitations:

  * per-epoch validation + load_best_model_at_end  -> reports the epoch-11-style
    best checkpoint instead of the overfit epoch-100 one
  * EarlyStoppingCallback                          -> stops once val loss stops
    improving, so you don't train 89 wasted epochs
  * --seeds 13 21 42                               -> multi-seed runs so you can
    report mean +/- std instead of a single number

This keeps your data format and MuRIL setup; edit the CONFIG block for your paths.
Requires: transformers, datasets, torch. (Left runnable on your machine; not run here.)

Run:
    python finetune_fixed.py --train train.json --val val.json --seeds 13 21 42
"""
import argparse, json, numpy as np
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, EarlyStoppingCallback,
                          default_data_collator, set_seed)
from datasets import Dataset

MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 384


def get_gold(ex):
    if "answers" in ex and ex["answers"]:
        a = ex["answers"]
        a = a[0] if isinstance(a, list) else a
        return a["text"], a.get("answer_start", ex["context"].find(a["text"]))
    txt = ex.get("answer", ex.get("answer_text", ""))
    return txt, ex.get("answer_start_char", ex.get("answer_start", ex["context"].find(txt)))


def build_dataset(path):
    raw = json.load(open(path, encoding="utf-8"))
    rows = []
    for ex in raw:
        text, start = get_gold(ex)
        if start is None or start < 0:
            start = ex["context"].find(text)
        rows.append({"question": ex["question"], "context": ex["context"],
                     "answer_text": text, "answer_start": start})
    return Dataset.from_list(rows)


def make_prepare(tok):
    def prepare(batch):
        enc = tok(batch["question"], batch["context"], truncation="only_second",
                  max_length=MAX_LEN, padding="max_length",
                  return_offsets_mapping=True)
        starts, ends = [], []
        for i, off in enumerate(enc["offset_mapping"]):
            a_start = batch["answer_start"][i]
            a_end = a_start + len(batch["answer_text"][i])
            seq = enc.sequence_ids(i)
            ctx_tok = [j for j, s in enumerate(seq) if s == 1]
            s_tok = e_tok = 0
            for j in ctx_tok:
                cs, ce = off[j]
                if cs <= a_start < ce:
                    s_tok = j
                if cs < a_end <= ce:
                    e_tok = j
            starts.append(s_tok); ends.append(max(e_tok, s_tok))
        enc["start_positions"] = starts
        enc["end_positions"] = ends
        enc.pop("offset_mapping")
        return enc
    return prepare


def run_one(seed, train_ds, val_ds, tok):
    set_seed(seed)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    args = TrainingArguments(
        output_dir=f"./out_seed{seed}",
        learning_rate=2e-5, weight_decay=0.01,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        num_train_epochs=40,                    # cap; early stopping ends it sooner
        warmup_ratio=0.06,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,            # <-- the key fix
        metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=2, logging_steps=50, report_to=[],
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=default_data_collator, tokenizer=tok,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    best = trainer.evaluate()
    trainer.save_model(f"./best_seed{seed}")
    return best["eval_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    prepare = make_prepare(tok)
    train_ds = build_dataset(args.train).map(prepare, batched=True,
               remove_columns=["question", "context", "answer_text", "answer_start"])
    val_ds = build_dataset(args.val).map(prepare, batched=True,
               remove_columns=["question", "context", "answer_text", "answer_start"])

    losses = [run_one(s, train_ds, val_ds, tok) for s in args.seeds]
    print(f"\nbest val loss per seed: {[round(x,4) for x in losses]}")
    print(f"mean +/- std          : {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
    print("Now run evaluate_fixed.py --model ./best_seed<S> on each seed and report")
    print("mean +/- std of F1/EM (per subset) in Table 3.")


if __name__ == "__main__":
    main()
