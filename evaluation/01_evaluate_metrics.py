"""
=============================================================================
Script  : 01_evaluate_metrics.py
Folder  : 04_evaluation/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 9
=============================================================================

PURPOSE
-------
This script evaluates the fine-tuned MuRIL model on the held-out test set
and computes two standard NLP metrics: F1 Score and Exact Match (EM).

It also runs the same evaluation on the BASELINE model (before fine-tuning)
so we can report the improvement in the paper as a before/after comparison.

WHAT ARE F1 AND EXACT MATCH?
-----------------------------
Both metrics compare the model's predicted answer to the correct answer.

EXACT MATCH (EM):
    1 if the prediction exactly matches the correct answer (after normalisation)
    0 otherwise.
    Strict — "7" and "7 apples" would score 0 even though both are right.

    Example:
        Correct   : "7"
        Predicted : "7"       → EM = 1  ✅
        Predicted : "7 cups"  → EM = 0  ❌

F1 SCORE:
    Measures word-level overlap between prediction and correct answer.
    More forgiving than EM — partial credit for partial matches.

    Formula:
        precision = (shared words) / (words in prediction)
        recall    = (shared words) / (words in correct answer)
        F1        = 2 × precision × recall / (precision + recall)

    Example:
        Correct   : "పది ఆపిల్లు"   (ten apples)   — 2 words
        Predicted : "పది"            (ten)          — 1 word
        Shared    : "పది"                           — 1 word
        Precision : 1/1 = 1.0
        Recall    : 1/2 = 0.5
        F1        : 2×1.0×0.5 / (1.0+0.5) = 0.667

WHY THESE METRICS?
    They are the standard metrics for SQuAD-style extractive QA and are
    used in virtually every QA paper. Reviewers expect to see them.
    Reporting both gives a complete picture — EM is strict, F1 is lenient.

NORMALISATION:
    Before comparing, we normalise both prediction and correct answer:
        - Lowercase
        - Remove punctuation
        - Strip extra whitespace
        - Remove articles (a, an, the) — not applicable to Telugu
    This makes comparison fair and consistent with SQuAD evaluation standards.

INPUTS
------
    data/telugu_math_test.json            ← test set from Week 7
    models/muril_telugu_math/             ← fine-tuned model from Week 8
    google/muril-base-cased               ← baseline (downloaded if needed)

OUTPUTS
-------
    Console : F1 and EM scores for baseline and fine-tuned model
    File    : models/evaluation_results.json  ← for the paper

HOW TO RUN
----------
    Terminal : python 04_evaluation/01_evaluate_metrics.py
    Notebook : run each cell

AUTHOR
------
    Research Scholar : Siddharth Kolakaluri
    Research Mentor  : TY
    Program          : Lumiere Education — Research Scholar Program
    Date             : 2025

=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import json
import os
import string
import re
from collections import Counter

# =============================================================================
# PATHS
# =============================================================================

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
except NameError:
    _REPO_ROOT  = os.path.expanduser(
        "~/Documents/lumiere_education/2026_siddharth/telugu_math_chatbot"
    )
    _SCRIPT_DIR = os.path.join(_REPO_ROOT, "04_evaluation")

TEST_PATH        = os.path.join(_REPO_ROOT, "data",   "telugu_math_test.json")
FINETUNED_MODEL  = os.path.join(_REPO_ROOT, "model",  "muril_telugu_math")
BASELINE_MODEL   = "google/muril-base-cased"
RESULTS_PATH     = os.path.join(_REPO_ROOT, "model",  "evaluation_results.json")

MAX_LENGTH = 384
MAX_ANSWER_LEN = 50   # maximum tokens in predicted answer span

# =============================================================================
# NORMALISATION AND METRIC FUNCTIONS
# =============================================================================

def normalise(text: str) -> str:
    """
    Normalise answer text before comparison.

    Applies the same normalisation used in official SQuAD evaluation:
        1. Lowercase everything
        2. Remove punctuation
        3. Collapse multiple spaces into one
        4. Strip leading/trailing whitespace

    This ensures fair comparison — "7." and "7" should score as equal,
    and "పది" and "పది " should also match.

    Args:
        text (str): Raw answer text.

    Returns:
        str: Normalised text.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation — string.punctuation covers !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match score between prediction and ground truth.

    Returns 1.0 if they match after normalisation, 0.0 otherwise.

    Args:
        prediction   (str): Model's predicted answer.
        ground_truth (str): Correct answer from dataset.

    Returns:
        float: 1.0 (match) or 0.0 (no match).
    """
    return float(normalise(prediction) == normalise(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score between prediction and ground truth.

    Splits both strings into tokens (words) and measures overlap.
    This is the standard SQuAD F1 computation.

    Args:
        prediction   (str): Model's predicted answer.
        ground_truth (str): Correct answer from dataset.

    Returns:
        float: F1 score between 0.0 and 1.0.
    """
    pred_tokens  = normalise(prediction).split()
    truth_tokens = normalise(ground_truth).split()

    # If either is empty, F1 is 0
    if not pred_tokens or not truth_tokens:
        return 0.0

    # Count shared tokens using Counter (handles duplicate words correctly)
    pred_counter  = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    # Shared = intersection of both token bags
    shared = sum((pred_counter & truth_counter).values())

    if shared == 0:
        return 0.0

    precision = shared / len(pred_tokens)
    recall    = shared / len(truth_tokens)
    f1        = 2 * precision * recall / (precision + recall)

    return f1


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def predict_answer(model, tokenizer, question: str, context: str, device: str) -> str:
    """
    Run inference on one QA example and return the predicted answer text.

    This is the same manual inference approach we used in Week 4's baseline
    script — no pipeline needed, works with all transformers versions.

    Steps:
        1. Tokenize question + context together
        2. Forward pass through model → start_logits, end_logits
        3. Find best start and end token positions
        4. Decode token span back to text

    Args:
        model     : Loaded QA model (fine-tuned or baseline).
        tokenizer : Loaded tokenizer.
        question  (str): Telugu question.
        context   (str): Telugu context passage.
        device    (str): "cuda", "mps", or "cpu".

    Returns:
        str: Predicted answer text extracted from context.
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors = "pt",
        truncation     = "only_second",
        max_length     = MAX_LENGTH,
        padding        = True,
        return_offsets_mapping = True,
    )

    offset_mapping = inputs.pop("offset_mapping")
    sequence_ids   = inputs.sequence_ids(0)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits   = outputs.end_logits[0]

    # Mask out non-context tokens so the answer can only come from context
    # sequence_ids == 1 means context tokens
    inf = float("-inf")
    start_logits = [
        s.item() if sequence_ids[i] == 1 else inf
        for i, s in enumerate(start_logits)
    ]
    end_logits = [
        e.item() if sequence_ids[i] == 1 else inf
        for i, e in enumerate(end_logits)
    ]

    # Find best valid (start, end) pair where start <= end
    # and span length <= MAX_ANSWER_LEN tokens
    best_score  = float("-inf")
    best_start  = 0
    best_end    = 0

    for start_idx, start_score in enumerate(start_logits):
        if start_score == inf:
            continue
        for end_idx in range(start_idx, min(start_idx + MAX_ANSWER_LEN, len(end_logits))):
            if end_logits[end_idx] == inf:
                continue
            score = start_score + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end   = end_idx

    # Decode the winning token span back to text
    answer_ids = inputs["input_ids"][0][best_start : best_end + 1]
    answer     = tokenizer.decode(answer_ids.cpu(), skip_special_tokens=True)

    return answer.strip()


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, tokenizer, test_data: list, device: str, label: str) -> dict:
    """
    Evaluate a model on the full test set and return F1 and EM scores.

    Runs inference on every test example, computes F1 and EM for each,
    then averages across all examples.

    Also prints per-example results for the first few examples so we
    can see qualitatively what the model is doing.

    Args:
        model     : Loaded QA model.
        tokenizer : Loaded tokenizer.
        test_data (list): Test examples from telugu_math_test.json.
        device    (str) : "cuda", "mps", or "cpu".
        label     (str) : "Baseline" or "Fine-tuned" — for printing.

    Returns:
        dict: {"f1": float, "em": float, "n": int, "per_example": list}
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {label}")
    print(f"{'=' * 60}")
    print(f"  Running inference on {len(test_data)} test examples...\n")

    model.eval()   # set model to evaluation mode — disables dropout

    f1_scores  = []
    em_scores  = []
    per_example = []

    for i, ex in enumerate(test_data):
        prediction   = predict_answer(
            model, tokenizer,
            ex["question"], ex["context"], device
        )
        ground_truth = ex["answer_text"]

        f1 = f1_score(prediction, ground_truth)
        em = exact_match(prediction, ground_truth)

        f1_scores.append(f1)
        em_scores.append(em)

        per_example.append({
            "id"         : ex.get("id", str(i)),
            "question"   : ex["question"],
            "correct"    : ground_truth,
            "predicted"  : prediction,
            "f1"         : round(f1, 4),
            "em"         : em,
            "source"     : ex.get("source", "unknown"),
        })

        # Print first 5 examples for qualitative inspection
        if i < 5:
            print(f"  Example {i + 1}:")
            print(f"    Question  : {ex['question'][:70]}...")
            print(f"    Correct   : {ground_truth}")
            print(f"    Predicted : {prediction if prediction else '(empty)'}")
            print(f"    F1={f1:.3f}  EM={em:.0f}")
            print()

        # Progress every 10 examples
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(test_data)}")

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0

    print(f"\n  ── {label} Results ──────────────────")
    print(f"  F1 Score    : {avg_f1 * 100:.1f}%")
    print(f"  Exact Match : {avg_em * 100:.1f}%")
    print(f"  Examples    : {len(test_data)}")

    return {
        "label"      : label,
        "f1"         : round(avg_f1, 4),
        "em"         : round(avg_em, 4),
        "f1_pct"     : round(avg_f1 * 100, 1),
        "em_pct"     : round(avg_em * 100, 1),
        "n"          : len(test_data),
        "per_example": per_example,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main evaluation pipeline.

    1. Load test data
    2. Detect device
    3. Evaluate BASELINE model (zero-shot MuRIL, no fine-tuning)
    4. Evaluate FINE-TUNED model (our trained model)
    5. Print comparison table
    6. Save results to JSON for the paper
    """
    print("=" * 60)
    print("Telugu Math Chatbot — Lumiere Research Project")
    print("Script: 01_evaluate_metrics.py  |  Week 9")
    print("=" * 60)

    # ── Load test data ──────────────────────────────────────────────────────
    print(f"\n⏳ Loading test data from: {TEST_PATH}")
    if not os.path.exists(TEST_PATH):
        print(f"❌ Test file not found. Run emergency_merge.py first.")
        return

    with open(TEST_PATH, encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test examples")

    # ── Device ──────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Device: {device}")

    # ── Load tokenizer once — shared by both models ──────────────────────────
    print(f"\n⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
    print(f"✅ Tokenizer ready")

    # ── EVALUATE BASELINE ────────────────────────────────────────────────────
    print(f"\n⏳ Loading baseline model (google/muril-base-cased)...")
    baseline_model = AutoModelForQuestionAnswering.from_pretrained(BASELINE_MODEL)
    baseline_model = baseline_model.to(device)

    baseline_results = evaluate_model(
        baseline_model, tokenizer, test_data, device, "Baseline (no fine-tuning)"
    )

    # Free baseline model memory before loading fine-tuned
    del baseline_model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # ── EVALUATE FINE-TUNED ──────────────────────────────────────────────────
    if not os.path.exists(FINETUNED_MODEL):
        print(f"\n⚠️  Fine-tuned model not found at: {FINETUNED_MODEL}")
        print(f"   Skipping fine-tuned evaluation.")
        finetuned_results = None
    else:
        print(f"\n⏳ Loading fine-tuned model from: {FINETUNED_MODEL}")
        finetuned_model = AutoModelForQuestionAnswering.from_pretrained(FINETUNED_MODEL)
        finetuned_model = finetuned_model.to(device)

        finetuned_results = evaluate_model(
            finetuned_model, tokenizer, test_data, device, "Fine-tuned MuRIL"
        )

    # ── COMPARISON TABLE ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS COMPARISON — copy this into your paper!")
    print(f"{'=' * 60}")
    print(f"\n  {'Model':<30} {'F1 Score':>10} {'Exact Match':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    print(f"  {'Baseline (MuRIL, no fine-tuning)':<30} "
          f"{baseline_results['f1_pct']:>9.1f}% "
          f"{baseline_results['em_pct']:>11.1f}%")
    if finetuned_results:
        print(f"  {'Fine-tuned MuRIL (ours)':<30} "
              f"{finetuned_results['f1_pct']:>9.1f}% "
              f"{finetuned_results['em_pct']:>11.1f}%")
        improvement_f1 = finetuned_results['f1_pct'] - baseline_results['f1_pct']
        improvement_em = finetuned_results['em_pct'] - baseline_results['em_pct']
        print(f"\n  Improvement (F1)         : {improvement_f1:+.1f}%")
        print(f"  Improvement (Exact Match): {improvement_em:+.1f}%")

    print(f"\n  Test set size : {len(test_data)} examples")
    print(f"  Training size : 368 examples (1 epoch on Apple MPS)")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────────
    results = {
        "baseline"   : baseline_results,
        "finetuned"  : finetuned_results,
        "test_size"  : len(test_data),
        "model_path" : FINETUNED_MODEL,
    }
    # Remove per_example from saved results to keep file small
    for key in ["baseline", "finetuned"]:
        if results[key]:
            results[key] = {k: v for k, v in results[key].items() if k != "per_example"}

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {RESULTS_PATH}")
    print(f"\n  Next step:")
    print(f"  → Week 10: python 05_chatbot/01_gradio_app.py")
    print(f"             Build and launch the interactive chatbot demo")


if __name__ == "__main__":
    main()
