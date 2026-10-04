"""
indicqa_offset_fix.py  (v2)
---------------------------
Rescues IndicQA Telugu examples that fail a span-consistency check, and reports
HOW each one was rescued so we can tell whether the "byte offset" explanation is
actually correct.

For every example it tries, in order:
  1. stored offset as a CHARACTER offset       -> "already_ok"
  2. stored offset as a BYTE offset, converted -> "recovered_byte"  (paper's theory)
  3. plain substring search for the answer     -> "recovered_find"  (works regardless
                                                   of the true offset bug)
Anything whose answer text isn't in the context at all is "broken".

If recovered_byte is large, the paper's byte-offset explanation holds. If it's tiny
but recovered_find is large, the byte-offset story is NOT what's happening, and
Sections 3.2 / 7.1 should be corrected to say the answers are recoverable by
substring search but the exact offset-corruption cause is undetermined.

Run:
    python3 fixes/indicqa_offset_fix.py --in data/indicqa_raw.te.json --out data/indicqa_te_recovered.json
"""
import argparse, json, sys


def byte_to_char_offset(context, byte_off):
    enc = context.encode("utf-8")
    if byte_off is None or byte_off > len(enc):
        return None
    try:
        return len(enc[:byte_off].decode("utf-8"))
    except UnicodeDecodeError:
        return None


def span_ok(context, answer_text, char_start):
    if char_start is None or char_start < 0 or not answer_text:
        return False
    return context[char_start:char_start + len(answer_text)] == answer_text


def iter_examples(raw):
    """Handle both the raw HuggingFace nested SQuAD form and a flat list."""
    if isinstance(raw, dict) and "data" in raw:
        for art in raw["data"]:
            for para in art.get("paragraphs", []):
                ctx = para["context"]
                for qa in para.get("qas", []):
                    answers = qa.get("answers") or []
                    if not answers:
                        yield {"id": qa.get("id"), "context": ctx,
                               "question": qa.get("question", ""),
                               "answer_text": "", "answer_start": None}
                        continue
                    ans = answers[0]
                    yield {"id": qa.get("id"), "context": ctx,
                           "question": qa.get("question", ""),
                           "answer_text": ans.get("text", ""),
                           "answer_start": ans.get("answer_start")}
    elif isinstance(raw, list):
        for ex in raw:
            yield {"id": ex.get("id"), "context": ex["context"],
                   "question": ex.get("question", ""),
                   "answer_text": ex.get("answer_text", ex.get("answer", "")),
                   "answer_start": ex.get("answer_start")}
    else:
        raise ValueError("Unrecognized IndicQA JSON shape.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    raw = json.load(open(args.inp, encoding="utf-8"))
    total = no_answer = already_ok = rec_byte = rec_find = broken = 0
    out = []

    for ex in iter_examples(raw):
        total += 1
        ctx, ans, raw_start = ex["context"], ex["answer_text"], ex["answer_start"]

        if not ans:
            no_answer += 1
            continue

        if span_ok(ctx, ans, raw_start):
            already_ok += 1; corrected = raw_start
        else:
            cs = byte_to_char_offset(ctx, raw_start)
            if span_ok(ctx, ans, cs):
                rec_byte += 1; corrected = cs
            else:
                idx = ctx.find(ans)
                if idx != -1:
                    rec_find += 1; corrected = idx
                else:
                    broken += 1; continue

        out.append({"id": ex["id"], "context": ctx, "question": ex["question"],
                    "answer_text": ans, "answer_start": corrected,
                    "source": "indicqa", "split": "all"})

    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    kept = already_ok + rec_byte + rec_find
    answerable = total - no_answer
    print("=" * 60)
    print(f"total examples read        : {total}")
    print(f"  unanswerable (no answer) : {no_answer}")
    print(f"  answerable               : {answerable}")
    print("-" * 60)
    print(f"already char-correct       : {already_ok}")
    print(f"recovered via BYTE->char   : {rec_byte}   <- paper's theory")
    print(f"recovered via substring    : {rec_find}   <- theory-independent")
    print(f"genuinely broken           : {broken}")
    print("-" * 60)
    print(f"KEPT (usable) : {kept}  ({kept/max(answerable,1)*100:.1f}% of answerable)")
    print(f"  vs the paper's 162")
    print("=" * 60)
    if rec_byte >= rec_find and rec_byte > 0:
        print("VERDICT: byte-offset explanation is supported.")
    elif rec_find > rec_byte:
        print("VERDICT: byte-offset explanation is NOT the main cause -")
        print("         most examples were rescued by substring search, not by the")
        print("         byte->char conversion. Correct Sections 3.2 / 7.1 accordingly.")
    else:
        print("VERDICT: inconclusive; inspect a few examples by hand.")


if __name__ == "__main__":
    sys.exit(main())
