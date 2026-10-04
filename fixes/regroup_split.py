"""
regroup_split.py
----------------
Re-splits the merged data so that examples sharing a context (or a MAWPS template)
never straddle train/test/val. This removes the leakage the audit found.

It pools your current train/val/test, groups examples by a context "signature"
(answer clause removed + numbers masked -- the same signature the audit uses),
shuffles whole GROUPS with seed 42, and assigns 80/10/10 by example count so that
no signature appears on two sides. Writes NEW files (…_grouped.json) so nothing is
overwritten, and prints a self-check of the resulting overlap.

Run:
    python3 fixes/regroup_split.py --train data/telugu_math_train.json \
        --val data/telugu_math_val.json --test data/telugu_math_test.json --outdir data
"""
import argparse, json, os, re, random


def signature(ctx):
    ctx = re.sub(r"(సమాధానం|The answer is)\s*-?\d+(?:[.,]\d+)?\.?", "", ctx)
    ctx = re.sub(r"-?\d+(?:[.,]\d+)?", "#", ctx)
    return " ".join(ctx.split()).lower()


def load(p):
    return json.load(open(p, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load(args.train) + load(args.val) + load(args.test)
    print(f"pooled examples: {len(pool):,}")

    # group by context signature
    groups = {}
    for ex in pool:
        groups.setdefault(signature(ex["context"]), []).append(ex)
    print(f"distinct context groups: {len(groups):,}")

    keys = list(groups.keys())
    random.seed(args.seed)
    random.shuffle(keys)

    total = len(pool)
    train, val, test = [], [], []
    for k in keys:
        if len(train) < 0.80 * total:
            train += groups[k]
        elif len(train) + len(val) < 0.90 * total:
            val += groups[k]
        else:
            test += groups[k]

    # report
    def brk(split):
        i = sum(1 for e in split if e.get("source") == "indicqa")
        m = len(split) - i
        return f"{len(split):,} (indicqa {i}, mawps {m})"
    print(f"\ntrain: {brk(train)}\nval  : {brk(val)}\ntest : {brk(test)}")

    # self-check: how many test signatures also appear in train?
    tr_sig = {signature(e['context']) for e in train}
    leaked = sum(1 for e in test if signature(e['context']) in tr_sig)
    print(f"\nself-check: test contexts whose signature is also in train: "
          f"{leaked} ({leaked/max(len(test),1)*100:.1f}%)  <- want ~0")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(args.outdir, f"telugu_math_{name}_grouped.json")
        json.dump(data, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"saved {out}: {len(data):,}")


if __name__ == "__main__":
    main()
