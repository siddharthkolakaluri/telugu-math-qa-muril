"""
evaluate_fixed.py
-----------------
Replaces the single headline F1/EM with the numbers needed to interpret the model:

  (a) a RULE-BASED baseline that returns the number after the answer cue
      ("సమాధానం" / "The answer is"). This is the control the paper is missing;
      if it matches or beats the model on MAWPS, the 90% "means" template-copying.
  (b) F1 and EM reported SEPARATELY for the MAWPS and IndicQA subsets, so genuine
      QA performance is not hidden by the templated majority.
  (c) the fine-tuned model's F1/EM on the same splits, for a like-for-like table.

Run:
    python evaluate_fixed.py --test test.json --model ../03_model/finetuned

`test.json` is your held-out split (list of SQuAD-style dicts with fields:
context, question, answers{text,answer_start} OR answer/answer_start, and ideally
a 'source' field == "mawps" | "indicqa"). If 'source' is absent, the script infers
it from the fixed MAWPS question ("సమాధానం ఏమిటి").
"""
import argparse, json, re, string, collections

ANSWER_CUES = ["సమాధానం", "The answer is", "సమాధానం "]
MAWPS_QUESTION_MARKERS = ["సమాధానం ఏమిటి", "what is the answer"]


# ---------- SQuAD metric primitives ----------
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0 or not p or not g:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em_score(pred, gold):
    return float(normalize(pred) == normalize(gold))


# ---------- helpers ----------
def get_gold(ex):
    if "answers" in ex and ex["answers"]:
        a = ex["answers"]
        return (a[0]["text"] if isinstance(a, list) else a["text"])
    return ex.get("answer", ex.get("answer_text", ""))


def get_source(ex):
    if ex.get("source"):
        return ex["source"].lower()
    q = (ex.get("question") or "").lower()
    return "mawps" if any(m in q for m in MAWPS_QUESTION_MARKERS) else "indicqa"


def rule_based_predict(context):
    """Return the number following the answer cue; else the last number in context."""
    for cue in ANSWER_CUES:
        i = context.rfind(cue)
        if i != -1:
            tail = context[i + len(cue):]
            m = re.search(r"-?\d+(?:[.,]\d+)?", tail)
            if m:
                return m.group(0)
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", context)
    return nums[-1] if nums else ""


def score_predictions(examples, predict_fn):
    buckets = collections.defaultdict(lambda: {"f1": 0.0, "em": 0.0, "n": 0})
    for ex in examples:
        gold = get_gold(ex)
        pred = predict_fn(ex)
        src = get_source(ex)
        for key in (src, "all"):
            buckets[key]["f1"] += f1_score(pred, gold)
            buckets[key]["em"] += em_score(pred, gold)
            buckets[key]["n"] += 1
    for k in buckets:
        n = max(buckets[k]["n"], 1)
        buckets[k]["f1"] = 100 * buckets[k]["f1"] / n
        buckets[k]["em"] = 100 * buckets[k]["em"] / n
    return buckets


def model_predictor(model_dir):
    """Lazy-load the fine-tuned QA model; returns a predict(ex)->str fn."""
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).eval()

    def predict(ex):
        enc = tok(ex["question"], ex["context"], return_tensors="pt",
                  truncation="only_second", max_length=384, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0]
        seq_ids = enc.sequence_ids(0)
        with torch.no_grad():
            out = model(**enc)
        start, end = out.start_logits[0].clone(), out.end_logits[0].clone()
        for i, s in enumerate(seq_ids):      # restrict to context tokens
            if s != 1:
                start[i] = end[i] = -1e9
        si = int(start.argmax())
        ei = int(end[si:si + 50].argmax()) + si
        cs, ce = int(offsets[si][0]), int(offsets[ei][1])
        return ex["context"][cs:ce].replace("##", "")
    return predict


def print_table(title, buckets):
    print(f"\n=== {title} ===")
    print(f"{'subset':<10}{'n':>6}{'F1':>10}{'EM':>10}")
    for k in ("mawps", "indicqa", "all"):
        if k in buckets:
            b = buckets[k]
            print(f"{k:<10}{b['n']:>6}{b['f1']:>10.1f}{b['em']:>10.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default=None, help="path to fine-tuned model dir")
    args = ap.parse_args()
    data = json.load(open(args.test, encoding="utf-8"))

    print_table("Rule-based baseline (number after cue)",
                score_predictions(data, lambda ex: rule_based_predict(ex["context"])))

    if args.model:
        pred = model_predictor(args.model)
        print_table("Fine-tuned MuRIL", score_predictions(data, pred))
    else:
        print("\n(no --model given; ran rule-based baseline only)")

    print("\nReport BOTH tables. If the rule-based MAWPS F1 is close to the model's,")
    print("state plainly that the MAWPS score reflects template-copying, and treat the")
    print("indicqa row as the real QA signal.")


if __name__ == "__main__":
    main()
