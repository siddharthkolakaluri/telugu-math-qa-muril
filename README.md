# Telugu math QA with fine-tuned MuRIL

Extractive question answering in Telugu for elementary mathematics. MuRIL is
fine-tuned on a 3,130-example corpus built from AI4Bharat IndicQA passages and
machine-translated MAWPS word problems, then evaluated on a leakage-free,
group-split test set.

This is the code and data behind a feasibility study. The headline result is a
negative one, and it is the point of the project: on the templated half of the
data a model scoring 100.0 F1 is matched exactly by a five-line rule that never
learned anything, so that score measures template-copying rather than
reasoning. Scores are therefore reported per subset throughout, never blended.

## What the model does, and what it cannot do

It selects a **span of the passage you give it**. MuRIL is an encoder: its
output head predicts a start and an end position over the input tokens, so it
can only return text that is already written in the passage.

It cannot compute. Give it *"Ramu has 10 apples, he gave away 3"* and ask how
many remain, and it cannot answer 7, because 7 appears nowhere in the input.
This is a property of the model class, not a training shortfall. As a tutoring
system it does not work, and demonstrating that precisely is what this
repository is for.

## Results

Leakage-free test set, 313 examples (167 MAWPS, 146 IndicQA). The rule-based
baseline returns the number following the answer cue.

| Model | Subset | F1 | Exact Match |
|---|---|---|---|
| Rule-based baseline | MAWPS | 100.0 | 100.0 |
| Rule-based baseline | IndicQA | 2.7 | 2.7 |
| Fine-tuned MuRIL | MAWPS | 100.0 | 100.0 |
| Fine-tuned MuRIL | IndicQA | **57.7** | **47.3** |

Two comparisons make these numbers readable. A trivial extractor also reaches
100.0 on MAWPS, so that subset is solved without arithmetic. On IndicQA the
same extractor manages 2.7, so the model's 57.7 is a real gain of roughly 55
points, attributable to learned span localisation.

The blended figure across both subsets is 80.3 F1 / 75.4 EM. It mixes a trivial
task with a genuine one and should not be quoted as a single score.

### Seed robustness

Three seeds, identical settings, same test set.

| Seed | IndicQA F1 | IndicQA EM | MAWPS F1 | Best epoch | Val loss | Train time |
|---|---|---|---|---|---|---|
| 42 | 57.7 | 47.3 | 100.0 | 8 | 1.1240 | reused |
| 123 | 62.0 | 52.1 | 100.0 | 6 | 1.1231 | 169.5 min |
| 7 | 57.7 | 48.6 | 100.0 | 6 | 1.1919 | 171.0 min |
| **mean** | **59.1 ± 2.5** | **49.3 ± 2.5** | 100.0 | | | |

Three seeds is a small sample for a standard deviation, and the IndicQA test
set is 146 examples, so read the interval as indicative rather than exact.

## A finding about IndicQA Telugu

Useful to anyone else using this dataset.

The `answer_start` offsets in the IndicQA Telugu split are unreliable: 1,572 of
1,734 examples (90.6%) fail a span-consistency check, where the substring at the
stored position does not match the stored answer string.

This was initially attributed to a byte-versus-character offset mismatch, on the
reasoning that Telugu characters occupy three UTF-8 bytes. **That explanation is
wrong.** Re-running the recovery over the full file:

| Outcome | Count |
|---|---|
| Already had correct character offsets | 165 |
| Recovered by locating the answer string in the context | 1,227 |
| Recovered by byte-to-character conversion | **0** |
| Total recovered | 1,392 of 1,398 (99.6%) |
| Unrecoverable, answer not verbatim in context | 6 |

Not a single example is fixed by byte conversion, so the byte-offset theory is
refuted rather than merely unsupported. The answer text is present in the
contexts regardless, so substring search recovers almost all of it.

**If you use IndicQA Telugu, validate spans by substring search rather than
trusting `answer_start`**, and compare against the official AI4Bharat loader
before attributing a discrepancy to the dataset. `fixes/indicqa_offset_fix.py`
does the recovery.

## Dataset

| Source | Raw | After cleaning | Language |
|---|---|---|---|
| AI4Bharat IndicQA | 1,734 | 1,392 (80.3%) | Telugu |
| MAWPS, translated | 1,772 | 1,738 (98.1%) | En → Te |
| Combined | 3,506 | 3,130 | Telugu |

`mawps_telugu_clean.json` holds all 1,772 translated problems. Deduplication to the 1,738 used in the corpus happens at the merge step, so the split files are the authoritative count.

Split 80/10/10 into 2,504 train, 313 validation, 313 test. The split is by
**context group**, not by individual example: IndicQA passages are shared across
several questions and MAWPS contexts are near-identical by construction, so an
example-level shuffle puts near-duplicates on both sides of the split.
`fixes/leakage_audit.py` checks for residual overlap.

MAWPS translation used the Google Translate API via `deep-translator`. That API
is non-deterministic, so the construction step is not bit-for-bit reproducible
and the frozen dataset is shipped here rather than regenerated.

## Layout

```
data/          dataset construction scripts and the frozen corpus
model/         baseline loading, fine-tuning, training log
evaluation/    metrics, training curves
fixes/         offset recovery, group re-split, leakage audit, multi-seed run
chatbot/       Gradio demo and a manual test set
results/       multi-seed results and logs
superseded/    earlier results and demo, kept for transparency
```

`superseded/` holds the original evaluation run (90.4 F1 on a 190-example set)
and the first demo script. Neither reflects current results — they are retained
so the corrections can be checked against what they replaced.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Trained and evaluated on an Apple Silicon MacBook using the MPS backend. CUDA
and CPU both work; CPU is slow.

## Reproducing the results

```bash
bash run_pipeline.sh
```

Recovery, merge, group re-split and leakage audit take a few minutes. Training
takes roughly 2.8 hours per seed. A seed whose model directory already exists is
only re-evaluated, so an existing run can be reused:

```bash
cp -r model/muril_telugu_math_v2 model/muril_seed42
```

For a quick sanity check over all three seeds before committing to a full run:

```bash
SMOKE=1 python3 fixes/run_multiseed.py
```

Results land in `results/multiseed_results.json`.

## Running the demo

```bash
python chatbot/03_gradio_app_v3.py
```

Opens at http://localhost:7860. Enter a Telugu passage and a question; the
passage is redisplayed with the model's selected span marked in place, which is
the honest view of what an extractive model does. `chatbot/test_prompts.md` has
a manual test set grouped by what each case probes, including cases the model
cannot answer at all.

Pass `--model` to point at a different checkpoint. If fine-tuned weights are not
found the app says so on screen rather than quietly loading the untrained base
model.

## Limitations

- **Task formulation.** The MAWPS subset ends with the answer stated, so
  locating it needs no arithmetic. Only the IndicQA subset measures genuine
  question answering.
- **Residual leakage risk.** Deduplication was on context for MAWPS and question
  for IndicQA. MAWPS contexts are structurally near-identical, so near-duplicate
  overlap across the split is possible and the audit should be run.
- **Translation quality.** Automated validation confirmed only that the answer
  number survives into the translated context, not that the surrounding Telugu
  is fluent or that the problem's meaning survived. A human review pass is
  outstanding.
- **Span score is not calibrated.** The product of start and end softmax
  probabilities is not a probability of correctness. Observed cases where a
  correct answer scored lower than an incorrect one.
- **No educational evaluation.** No testing with Telugu-speaking students, and
  no analysis of the risk of confidently returning a wrong answer to a child.
- **Sample size.** Three seeds, 146 IndicQA test examples.

## Model weights

Not in this repository — the checkpoint is roughly 950 MB, past GitHub's file
size limit.

*(Add the Hugging Face link here once the weights are uploaded.)*

## Credits

- [MuRIL](https://huggingface.co/google/muril-base-cased), Google
- [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA), AI4Bharat
- [MAWPS](https://huggingface.co/datasets/mwpt5/MAWPS)
- Translation via the Google Translate API through `deep-translator`

Built as part of the Lumiere Education Research Scholar Program.

## License

MIT. See `LICENSE`.
