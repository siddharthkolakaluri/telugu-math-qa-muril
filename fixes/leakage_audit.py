"""
leakage_audit.py
----------------
Checks whether templated MAWPS contexts create train/test overlap. Exact-dedup on
context is not enough when contexts are near-identical ("Mary is baking a cake ...
The answer is 6." vs "... The answer is 8."). This reports:

  * exact context duplicates across splits
  * near-duplicate contexts across splits (answer stripped + high token Jaccard)

Run:
    python leakage_audit.py --train train.json --test test.json [--val val.json]

If near-duplicate overlap is non-trivial, re-split by GROUP (e.g. by MAWPS problem
template) so structurally identical items stay on one side, then re-evaluate.
"""
import argparse, json, re


def load(p):
    return json.load(open(p, encoding="utf-8"))


def strip_answer(ctx):
    # remove the trailing answer clause so templates with different numbers collide
    ctx = re.sub(r"(సమాధానం|The answer is)\s*-?\d+(?:[.,]\d+)?\.?", "", ctx)
    ctx = re.sub(r"-?\d+(?:[.,]\d+)?", "#", ctx)   # mask remaining numbers
    return " ".join(ctx.split()).lower()


def toks(s):
    return set(strip_answer(s).split())


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--thresh", type=float, default=0.9)
    args = ap.parse_args()

    train = load(args.train)
    test = load(args.test)
    tr_ctx = [e["context"] for e in train]
    te_ctx = [e["context"] for e in test]

    # exact (answer-stripped) duplicates
    tr_norm = {strip_answer(c) for c in tr_ctx}
    exact = sum(1 for c in te_ctx if strip_answer(c) in tr_norm)

    # near-duplicates by token Jaccard
    tr_tok = [toks(c) for c in tr_ctx]
    near = 0
    for c in te_ctx:
        ct = toks(c)
        if any(jaccard(ct, t) >= args.thresh for t in tr_tok):
            near += 1

    n = max(len(te_ctx), 1)
    print(f"test examples                         : {len(te_ctx)}")
    print(f"answer-stripped EXACT dup in train    : {exact}  ({exact/n*100:.1f}%)")
    print(f"NEAR-duplicate (Jaccard>={args.thresh}) in train: {near}  ({near/n*100:.1f}%)")
    if near / n > 0.2:
        print("\nWARNING: substantial template overlap across splits. The reported test")
        print("score is partly measuring memorized templates. Re-split by problem group.")
    else:
        print("\nOverlap looks limited, but still report these numbers in Section 7.2.")


if __name__ == "__main__":
    main()
