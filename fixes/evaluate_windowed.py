"""
evaluate_windowed.py
--------------------
Honest evaluation of the fine-tuned model on the leakage-free grouped test set.

For each question it runs the model over EVERY sliding window of the context and
keeps the most confident answer span across all of them (standard SQuAD-style
decoding). This is what lets long IndicQA contexts be scored fairly -- the old
single-window evaluator could not see answers past the first 384 tokens.

Reports F1 and Exact Match split by source (mawps vs indicqa), plus a rule-based
baseline (number after the answer cue) for comparison.

Run:
    python3 fixes/evaluate_windowed.py
"""
import json, os, re, string, collections, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

ROOT = os.path.expanduser("~/Documents/nlp_project")
MODEL_DIR = os.path.join(ROOT, "model", "muril_telugu_math_v2")
TEST_PATH = os.path.join(ROOT, "data", "telugu_math_test_grouped.json")
MAX_LEN, DOC_STRIDE = 384, 128
N_BEST, MAX_ANS_TOK = 20, 30
ANSWER_CUES = ["సమాధానం", "The answer is"]
MAWPS_Q = ["సమాధానం ఏమిటి", "what is the answer"]


# ---------- SQuAD metrics ----------
def normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
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
            if m:
                return m.group(0)
    nums = re.findall(r"-?\d+(?:[.,]\d+)?", context)
    return nums[-1] if nums else ""


# ---------- windowed model prediction ----------
def build_model_predictor():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR).eval()
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
           else "cpu")
    model.to(dev)

    def predict(question, context):
        enc = tok(question, context, max_length=MAX_LEN, truncation="only_second",
                  stride=DOC_STRIDE, return_overflowing_tokens=True,
                  return_offsets_mapping=True, padding="max_length",
                  return_tensors="pt")
        n = enc["input_ids"].shape[0]
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(dev),
                        attention_mask=enc["attention_mask"].to(dev),
                        token_type_ids=enc.get("token_type_ids").to(dev)
                        if "token_type_ids" in enc else None)
        starts = out.start_logits.cpu().numpy()
        ends = out.end_logits.cpu().numpy()
        best_text, best_score = "", -1e30
        for w in range(n):
            seq = enc.sequence_ids(w)
            offs = enc["offset_mapping"][w].tolist()
            s_logit, e_logit = starts[w], ends[w]
            ctx_idx = [i for i, sid in enumerate(seq) if sid == 1]
            if not ctx_idx:
                continue
            s_top = np.argsort(s_logit)[-N_BEST:][::-1]
            e_top = np.argsort(e_logit)[-N_BEST:][::-1]
            for si in s_top:
                if si not in ctx_idx:
                    continue
                for ei in e_top:
                    if ei not in ctx_idx or ei < si or ei - si + 1 > MAX_ANS_TOK:
                        continue
                    score = s_logit[si] + e_logit[ei]
                    if score > best_score:
                        cs, ce = offs[si][0], offs[ei][1]
                        best_score = score
                        best_text = context[cs:ce]
                    break  # first valid end for this start is the highest-scoring
        return best_text.strip()
    return predict


def score(examples, predict_fn):
    buckets = collections.defaultdict(lambda: {"f1": 0.0, "em": 0.0, "n": 0})
    samples = []
    for k, ex in enumerate(examples):
        gold = get_gold(ex)
        pred = predict_fn(ex)
        src = get_source(ex)
        for key in (src, "all"):
            buckets[key]["f1"] += f1_score(pred, gold)
            buckets[key]["em"] += em_score(pred, gold)
            buckets[key]["n"] += 1
        if src == "indicqa" and len(samples) < 4:
            samples.append((ex["question"][:50], gold, pred))
    for key in buckets:
        n = max(buckets[key]["n"], 1)
        buckets[key]["f1"] = 100 * buckets[key]["f1"] / n
        buckets[key]["em"] = 100 * buckets[key]["em"] / n
    return buckets, samples


def show(title, buckets):
    print(f"\n=== {title} ===")
    print(f"{'subset':<10}{'n':>6}{'F1':>9}{'EM':>9}")
    for k in ("mawps", "indicqa", "all"):
        if k in buckets:
            b = buckets[k]
            print(f"{k:<10}{b['n']:>6}{b['f1']:>9.1f}{b['em']:>9.1f}")


def main():
    data = json.load(open(TEST_PATH, encoding="utf-8"))
    print(f"test examples: {len(data)}   model: {MODEL_DIR}")

    print("\nrunning rule-based baseline...")
    rb, _ = score(data, lambda ex: rule_based(ex["context"]))
    show("Rule-based baseline (number after cue)", rb)

    print("\nrunning fine-tuned model (windowed)...")
    predict = build_model_predictor()
    mb, samples = score(data, lambda ex: predict(ex["question"], ex["context"]))
    show("Fine-tuned MuRIL v2", mb)

    print("\n--- sample IndicQA predictions ---")
    for q, gold, pred in samples:
        print(f"Q: {q}\n  gold: {gold}\n  pred: {pred}")

    print("\nReport the MAWPS and IndicQA rows separately in Table 3. The IndicQA")
    print("row is your genuine QA number; the MAWPS row is template-copying (compare")
    print("it to the rule-based baseline).")


if __name__ == "__main__":
    main()
