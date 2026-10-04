# Fixes for the empirical limitations

These four scripts address the limitations that can't be fixed by editing prose —
they need re-running on your model and data. Each maps to a limitation in Section 7.2
of the revised paper. Run them, then paste the real numbers back into the paper where
it currently says "to be performed" / "released code produces this".

| Script | Fixes | Produces |
|---|---|---|
| `indicqa_offset_fix.py` | IndicQA data loss (kept only 162/1734) | recovered dataset + true recovery count |
| `evaluate_fixed.py` | meaningless baseline; no subset split | rule-based baseline + MAWPS/IndicQA F1 & EM |
| `leakage_audit.py` | possible train/test leakage | exact + near-duplicate overlap across splits |
| `finetune_fixed.py` | epoch-100 overfit; single seed | early-stopped best checkpoint, multi-seed mean±std |

## Suggested order

1. **Recover IndicQA** (rebalances the dataset away from the template):
   ```
   python indicqa_offset_fix.py --in indicqa.te.json --out indicqa_te_recovered.json
   ```
   If "kept" is far above 162, the 9.34% retention was a discard artifact. Update
   Table 1 and Section 3.2 / 7.1 with the real number, and cross-check against the
   official AI4Bharat loader before attributing the offset issue to IndicQA.

2. **Re-merge & re-split** with the recovered IndicQA, then **audit leakage**:
   ```
   python leakage_audit.py --train train.json --test test.json --val val.json
   ```
   If near-duplicate overlap is high, re-split by MAWPS problem group.

3. **Re-train with early stopping + multiple seeds**:
   ```
   python finetune_fixed.py --train train.json --val val.json --seeds 13 21 42
   ```

4. **Evaluate with the baseline and per-subset breakdown**:
   ```
   python evaluate_fixed.py --test test.json --model ./best_seed42
   ```
   Report the rule-based row next to the model row. Expect the rule-based extractor to
   be competitive on MAWPS — that's the point, and stating it is what makes the paper
   honest. The IndicQA row is your real QA signal.

## What to expect

The headline "90.0% F1" will likely split into a high (template-copying) MAWPS number
that a regex nearly matches, and a much lower IndicQA number on the recovered genuine
QA. That is the accurate picture the revised Section 7.2 already prepares the reader for.

## Notes

- Scripts assume SQuAD-style examples: `context`, `question`, and either
  `answers:[{text, answer_start}]` or `answer`/`answer_text` (+ optional `source` and
  `answer_start_char`). They degrade gracefully if `source` is missing (inferred from
  the fixed MAWPS question).
- `finetune_fixed.py` / `evaluate_fixed.py` need `transformers`, `datasets`, `torch`.
- None of these were run for you — they need your weights/data and network access to
  the model hub, so the paper still reports your original epoch-100 numbers until you
  regenerate them.
