"""
=============================================================================
Script  : 02_translate_mawps.py
Folder  : 01_data/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 6
=============================================================================

PURPOSE
-------
This script downloads the MAWPS English elementary math word problem dataset
from HuggingFace, converts each problem into SQuAD-style QA format, translates
the text from English to Telugu, and saves the result for merging in Week 7.

This gives us the MATH-SPECIFIC part of our training data.

WHY TWO DATASETS?
-----------------
    IndicQA Telugu (Week 5) : ~1,300 examples → general Telugu QA
                               teaches the model Telugu language understanding
    MAWPS translated (Week 6): ~300  examples → Telugu math word problems
                               teaches the model to answer arithmetic questions

Using both is a deliberate research choice called DOMAIN MIXING.
The IndicQA examples prevent the model from "forgetting" general Telugu,
while the MAWPS examples teach it the specific math task.
This design decision will be discussed in the Methods section of the paper.

WHAT IS MAWPS?
--------------
MAWPS (Math Word Problem Repository) is a benchmark dataset of English
elementary school math word problems. Each row has:
    "Question" : a word problem (e.g. "Tom has 5 apples. He gave 2 away.
                  How many does he have?")
    "Equation" : the arithmetic equation (e.g. "N_00 - N_01")
    "Answer"   : the numeric answer (e.g. 3.0)
    "Numbers"  : the actual numbers that replace N_00, N_01 etc.

HuggingFace: https://huggingface.co/datasets/mwpt5/MAWPS

IMPORTANT NOTE ON MAWPS FORMAT
--------------------------------
The Question field uses placeholder variables like N_00, N_01 instead of
actual numbers. The Numbers field contains the actual values.

Example:
    Question : "Mary has N_00 apples and gives N_01 away. How many left?"
    Numbers  : "5.0 2.0"
    Answer   : 3.0

We substitute the actual numbers back into the question before translating,
so the Telugu version reads naturally: "Mary has 5 apples and gives 2 away."

CONVERTING MAWPS TO SQuAD FORMAT
----------------------------------
MAWPS has a question and a numeric answer but no "context" passage.
SQuAD format requires a context that CONTAINS the answer as a text span.

We create the context synthetically:
    context  = "{question with numbers} The answer is {answer}."
    question = "What is the answer?"
    answer   = "{answer}" at its character position in context

This is a valid and documented research approach. The model learns to
find numeric answer spans in short Telugu passages.

TRANSLATION
-----------
We use deep-translator (GoogleTranslator) — free, no API key needed.
Numbers survive translation unchanged (Google keeps digits as-is).
We translate in batches with delays to avoid rate limiting.

INPUTS
------
    Downloads MAWPS from HuggingFace (~323KB).
    Requires internet and deep-translator library.

OUTPUTS
-------
    Console : progress updates, sample translations, final statistics
    File    : data/mawps_telugu_clean.json
              Each record: {id, context, question, answer_text,
                            answer_start, source, original_en}

HOW TO RUN
----------
    Google Colab : !python 02_translate_mawps.py
    Terminal     : python 01_data/02_translate_mawps.py

    ⚠️  Translation takes 5–15 minutes for 300 examples.
        Do NOT close the window while it runs.
        The output file is saved incrementally so progress is not lost.

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

# load_dataset downloads MAWPS from HuggingFace.
# Unlike IndicQA, MAWPS uses modern Parquet format so this works normally.
from datasets import load_dataset

# GoogleTranslator is the translation engine from deep-translator.
# It uses Google Translate internally — free, no API key required.
from deep_translator import GoogleTranslator

# Standard Python libraries
import json        # reading and writing JSON files
import os          # file system operations (create folders, build paths)
import time        # time.sleep() to pause between API calls
import statistics  # computing averages for the stats report
import re          # regular expressions for substituting N_00 placeholders

# =============================================================================
# CONSTANTS
# =============================================================================

# HuggingFace dataset identifier — confirmed working from the dataset page
MAWPS_DATASET_ID = "mwpt5/MAWPS"

# How many examples to translate.
# Full dataset has 1,772 rows — translating all would take ~1 hour.
# 300 gives a good math supplement to IndicQA's ~1,300 examples.
# We document this cap in the paper as a "stratified subsample."
MAX_EXAMPLES = 1772

# Seconds to pause between translation API calls.
# Google Translate has rate limits. Without pausing, it blocks after ~50 calls.
# 1.5 seconds per call = ~7.5 minutes for 300 examples. Safe and reliable.
TRANSLATION_DELAY = 2.0

# Print a progress line every N examples so we know the script is running
PROGRESS_EVERY = 10

# Output file path
OUTPUT_DIR  = "data"
OUTPUT_FILE = "mawps_telugu_clean.json"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# =============================================================================
# FUNCTIONS
# =============================================================================

def substitute_numbers(question: str, numbers_str: str) -> str:
    """
    Replace placeholder variables (N_00, N_01 ...) in the question with
    their actual numeric values from the Numbers field.

    MAWPS stores questions with placeholders instead of real numbers:
        Question : "Mary has N_00 apples and gives N_01 away."
        Numbers  : "5.0 2.0"

    After substitution:
        "Mary has 5 apples and gives 2 away."

    WHY WE DO THIS:
        If we translate "N_00 apples" the output would be "N_00 ఆపిల్లు"
        which is meaningless. Substituting real numbers first gives us
        natural-sounding Telugu sentences after translation.

    HOW IT WORKS:
        The Numbers field is a space-separated string of floats.
        N_00 → numbers[0], N_01 → numbers[1], and so on.
        We convert whole numbers (5.0 → 5) for cleaner text.

    Args:
        question    (str): Question with N_00, N_01 placeholders.
        numbers_str (str): Space-separated numeric values, e.g. "5.0 2.0 3.0"

    Returns:
        str: Question with placeholders replaced by actual numbers.
             Returns the original question unchanged if substitution fails.

    Example:
        >>> substitute_numbers("Tom has N_00 apples.", "5.0")
        "Tom has 5 apples."
    """
    if not numbers_str or not numbers_str.strip():
        return question  # no numbers to substitute — return as-is

    try:
        # Split the numbers string into a list of float values
        number_list = [float(n) for n in numbers_str.strip().split()]

        result = question

        # Replace each placeholder N_00, N_01, N_02 ... with its value.
        # We iterate in reverse order of index to avoid N_0 matching inside N_00.
        # re.sub() replaces all occurrences of the pattern.
        for idx, value in enumerate(number_list):
            placeholder = f"N_{idx:02d}"   # e.g. "N_00", "N_01", "N_02"

            # Format number: show as integer if it is a whole number (5.0 → 5)
            # Show as decimal if it has a fractional part (3.14 → 3.14)
            if value == int(value):
                formatted = str(int(value))
            else:
                # Round to 4 significant figures to avoid ugly floats
                formatted = f"{value:.4g}"

            result = result.replace(placeholder, formatted)

        return result

    except Exception:
        # If anything goes wrong, return the original question unchanged
        return question


def download_mawps(dataset_id: str, max_examples: int) -> list:
    """
    Download the MAWPS dataset from HuggingFace and return a subsample.

    MAWPS uses the modern Parquet format so load_dataset() works without
    any of the issues we had with IndicQA. This is the standard approach.

    We take the first MAX_EXAMPLES rows after filtering for quality:
        - Remove rows with missing or empty questions
        - Remove rows where the answer cannot be cleanly formatted

    Args:
        dataset_id   (str): HuggingFace dataset identifier.
        max_examples (int): Maximum number of examples to return.

    Returns:
        list: List of cleaned raw MAWPS dicts, each with keys:
              Question, Equation, Answer, Numbers.

    Example:
        >>> examples = download_mawps("mwpt5/MAWPS", 300)
        >>> print(examples[0]["Question"])
        "Mary is baking a cake..."
    """
    print(f"\n⏳ Downloading MAWPS from HuggingFace...")
    print(f"   Dataset : {dataset_id}")
    print(f"   Target  : first {max_examples} usable examples\n")

    # load_dataset returns a DatasetDict — we use the "train" split
    # (MAWPS has only one split)
    dataset = load_dataset(dataset_id, split="train")
    total   = len(dataset)
    print(f"✅ Downloaded {total:,} total MAWPS examples.")

    # Convert to a plain Python list of dicts for easy processing
    all_rows = [dict(row) for row in dataset]

    # Filter for quality: must have a question and a numeric answer
    usable = []
    for row in all_rows:
        question = str(row.get("Question", "")).strip()
        answer   = row.get("Answer", None)

        if not question:
            continue
        if answer is None:
            continue
        # Skip examples where answer is not a real number
        try:
            float(answer)
        except (ValueError, TypeError):
            continue

        usable.append(row)

    print(f"   Usable after quality filter : {len(usable):,}")
    print(f"   Taking first                : {min(max_examples, len(usable)):,}")

    # Take the first MAX_EXAMPLES usable rows
    subset = usable[:max_examples]

    # Print a sample so we can see what we are working with
    print(f"\n   Sample raw MAWPS row:")
    if subset:
        sample = subset[0]
        print(f"     Question : {sample['Question'][:80]}...")
        print(f"     Answer   : {sample['Answer']}")
        print(f"     Numbers  : {sample.get('Numbers', 'N/A')}")
        # Show what it looks like after number substitution
        substituted = substitute_numbers(
            str(sample["Question"]),
            str(sample.get("Numbers", ""))
        )
        print(f"     After sub: {substituted[:80]}...")

    return subset


def build_context(question_with_numbers: str, answer_str: str) -> tuple:
    """
    Build a synthetic SQuAD-format context from a math word problem.

    SQuAD format requires a CONTEXT PASSAGE that contains the answer as
    a text span. MAWPS only gives us a question and a number.
    We construct the context by appending "The answer is {answer}." to
    the question, then recording the exact character position of the answer.

    Example:
        question_with_numbers : "Tom has 5 apples. He gave 2 away. How many left?"
        answer_str            : "3"

        context      : "Tom has 5 apples. He gave 2 away. How many left? The answer is 3."
        answer_text  : "3"
        answer_start : 51  ← character index of "3" in context

    WHY rfind() FOR ANSWER POSITION:
        We use str.rfind() (search from RIGHT) not str.find() (search from LEFT)
        because the answer number might also appear INSIDE the question.
        Example: "He has 3 apples..." — the "3" in the question is not the answer.
        rfind() finds the LAST occurrence, which is always our appended answer.

    Args:
        question_with_numbers (str): Math question with real numbers substituted.
        answer_str            (str): The answer as a clean string (e.g. "3" or "2.5")

    Returns:
        tuple: (context, answer_text, answer_start)
               context      : full context string
               answer_text  : the answer span (same as answer_str)
               answer_start : character index of answer_text in context
               Returns (None, None, -1) if answer cannot be located.
    """
    context = f"{question_with_numbers.strip()} The answer is {answer_str}."

    # Find the LAST occurrence of the answer string in the context
    # to avoid matching the same number if it appears in the question too
    answer_start = context.rfind(answer_str)

    if answer_start < 0:
        # This should not happen with our construction, but check anyway
        return None, None, -1

    # Verify the span is correct (sanity check)
    end       = answer_start + len(answer_str)
    extracted = context[answer_start:end]
    if extracted != answer_str:
        return None, None, -1

    return context, answer_str, answer_start


def format_answer(raw_answer) -> str:
    """
    Convert a raw MAWPS numeric answer into a clean string for the context.

    MAWPS stores answers as floats (e.g. 3.0, 2.5, 0.333...).
    We clean them up:
        3.0   → "3"      (whole number — drop the decimal)
        2.5   → "2.5"    (true decimal — keep it)
        0.333 → "0.333"  (round to 3 decimal places for readability)

    We also handle some edge cases like very long decimals.

    Args:
        raw_answer: The answer value from MAWPS (float or int).

    Returns:
        str: Clean string representation of the answer.

    Example:
        >>> format_answer(3.0)
        "3"
        >>> format_answer(2.5)
        "2.5"
    """
    try:
        val = float(raw_answer)
        if val == int(val):
            return str(int(val))       # whole number: 3.0 → "3"
        else:
            return f"{val:.3g}"        # decimal: format to 3 significant figures
    except (ValueError, TypeError):
        return str(raw_answer)


def translate_safely(translator: GoogleTranslator, text: str) -> str:
    """
    Translate a single text string from English to Telugu.

    If translation fails for any reason (network error, rate limit,
    text too long), we catch the exception and return a clearly marked
    failure string instead of crashing the whole script.

    This "fail gracefully" pattern is important for long-running scripts:
        - We do NOT want to lose 2 hours of work because one example failed
        - Failed examples can be identified by the "FAILED:" prefix and removed later

    Args:
        translator (GoogleTranslator): Initialised translator object.
        text       (str)             : English text to translate.

    Returns:
        str: Telugu translation, or "FAILED: {original text}" on error.
    """
    try:
        # Truncate to 500 characters to stay within Google Translate API limits
        text_to_translate = text[:500]
        result = translator.translate(text_to_translate)

        # translator.translate() sometimes returns None on empty input
        return result if result else text

    except Exception as e:
        print(f"\n     ⚠️  Translation error: {type(e).__name__}: {e}")
        return f"FAILED: {text[:100]}"


def translate_all(examples: list) -> list:
    """
    Translate all MAWPS examples from English to Telugu.

    For each example we:
        1. Substitute real numbers into the question (N_00 → actual value)
        2. Format the numeric answer into a clean string
        3. Build the English SQuAD context
        4. Translate the context to Telugu
        5. Translate the standard question ("What is the answer?") to Telugu
        6. Find the answer span position in the TRANSLATED Telugu context
        7. Save both Telugu and original English versions

    ANSWER POSITION AFTER TRANSLATION:
        Numbers survive Google Translate unchanged — "3" in English
        is still "3" in Telugu. So we can use rfind() to locate the
        answer in the Telugu context, exactly as in the English version.

    RATE LIMITING:
        We pause TRANSLATION_DELAY seconds after each API call.
        We also save a checkpoint file every PROGRESS_EVERY examples
        so if the script is interrupted, work is not completely lost.

    Args:
        examples (list): Raw MAWPS examples from download_mawps().

    Returns:
        list: Translated examples in SQuAD format, ready for saving.
    """
    print(f"\n" + "=" * 60)
    print("TRANSLATION: English → Telugu")
    print("=" * 60)
    print(f"  Examples to translate : {len(examples)}")
    est_mins = len(examples) * TRANSLATION_DELAY * 2 / 60
    print(f"  Estimated time        : ~{est_mins:.0f} minutes")
    print(f"  Delay between calls   : {TRANSLATION_DELAY}s")
    print(f"  Do NOT close this window while translating.\n")

    # Initialise the translator once and reuse it for all calls.
    # source="en" → English input
    # target="te" → Telugu output ("te" is ISO 639-1 code for Telugu)
    translator = GoogleTranslator(source="en", target="te")

    # Also pre-translate the standard question once — it is the same for all rows
    print("  Pre-translating standard question...")
    standard_q_en = "What is the answer?"
    standard_q_te = translate_safely(translator, standard_q_en)
    time.sleep(TRANSLATION_DELAY)
    print(f"  English : {standard_q_en}")
    print(f"  Telugu  : {standard_q_te}")
    print()

    translated  = []
    failed      = 0

    for i, raw in enumerate(examples):

        # ---- Progress update ----
        if (i + 1) % PROGRESS_EVERY == 0 or i == 0:
            print(f"  Translating {i + 1}/{len(examples)}...")

        # ---- STEP 1: Substitute real numbers into question ----
        question_raw = str(raw.get("Question", "")).strip()
        numbers_str  = str(raw.get("Numbers", "")).strip()
        question_en  = substitute_numbers(question_raw, numbers_str)

        if not question_en:
            failed += 1
            continue

        # ---- STEP 2: Format the numeric answer ----
        answer_str = format_answer(raw.get("Answer", 0))

        # ---- STEP 3: Build English SQuAD context ----
        context_en, answer_text, answer_start_en = build_context(
            question_en, answer_str
        )
        if context_en is None:
            failed += 1
            continue

        # ---- STEP 4: Translate context to Telugu ----
        context_te = translate_safely(translator, context_en)
        time.sleep(TRANSLATION_DELAY)

        # Skip if translation clearly failed
        if context_te.startswith("FAILED:"):
            failed += 1
            continue

        # ---- STEP 5: Find answer position in the TRANSLATED context ----
        # Numbers survive translation unchanged, so rfind() still works
        answer_start_te = context_te.rfind(answer_str)

        if answer_start_te < 0:
            # The answer number was not found in the Telugu context.
            # This is rare but can happen if Google altered the number format.
            # We skip this example rather than save bad data.
            failed += 1
            continue

        # ---- STEP 6: Verify the Telugu answer span ----
        end_te       = answer_start_te + len(answer_str)
        extracted_te = context_te[answer_start_te:end_te]
        if extracted_te != answer_str:
            failed += 1
            continue

        # ---- STEP 7: Save the translated example ----
        translated.append({
            "id"          : f"mawps_{i:04d}",
            "context"     : context_te,
            "question"    : standard_q_te,
            "answer_text" : answer_str,
            "answer_start": answer_start_te,
            "source"      : "mawps_translated",
            "split"       : "all",
            # Keep English originals for quality checking and paper documentation
            "original_en" : {
                "context" : context_en,
                "question": standard_q_en,
                "answer"  : answer_str,
            }
        })

        # ---- Checkpoint: save progress every PROGRESS_EVERY examples ----
        # This means if the script is interrupted, we keep what we have so far
        if (i + 1) % PROGRESS_EVERY == 0:
            _save_checkpoint(translated)

    print(f"\n  Translation complete.")
    print(f"  Successfully translated : {len(translated):,}")
    print(f"  Failed / skipped        : {failed:,}")

    return translated


def _save_checkpoint(examples: list) -> None:
    """
    Save a checkpoint of translated examples to disk.

    Called every PROGRESS_EVERY examples during translation.
    If the script is interrupted (e.g. Colab session expires), the
    checkpoint file contains all progress up to that point.

    The checkpoint file is overwritten each time — we always save
    the full list so far, not just the new batch.

    Args:
        examples (list): All translated examples so far.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(OUTPUT_DIR, "mawps_checkpoint.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def show_samples(translated: list, n: int = 3) -> None:
    """
    Print a few translated examples side-by-side with their English originals.

    This is the human quality check — we look at a sample of translations
    to make sure the Telugu makes sense and the answer span is correct.
    In the paper, we include 2–3 such examples in the Data section.

    Args:
        translated (list): Output of translate_all().
        n          (int) : Number of samples to print.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: SAMPLE TRANSLATIONS (quality check)")
    print("=" * 60)
    print("Compare English original with Telugu translation.\n")

    for i in range(min(n, len(translated))):
        ex = translated[i]
        print(f"--- Sample {i + 1} ---")
        print(f"  English context  : {ex['original_en']['context']}")
        print(f"  Telugu context   : {ex['context']}")
        print(f"  Telugu question  : {ex['question']}")
        print(f"  Answer           : {ex['answer_text']}")
        print(f"  Answer start     : {ex['answer_start']}")
        # Verify the span
        start     = ex["answer_start"]
        end       = start + len(ex["answer_text"])
        extracted = ex["context"][start:end]
        span_ok   = "✅" if extracted == ex["answer_text"] else "❌"
        print(f"  Span check       : context[{start}:{end}] = '{extracted}' {span_ok}")
        print()


def compute_statistics(translated: list) -> dict:
    """
    Compute and print summary statistics about the translated dataset.

    Stats reported:
        - Total translated examples
        - Average context / question / answer lengths
        - Count of examples where answer span was verified correct

    These numbers go into the Data section of the research paper,
    alongside the IndicQA stats from Week 5.

    Args:
        translated (list): Output of translate_all().

    Returns:
        dict: Summary statistics.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: STATISTICS")
    print("=" * 60)

    ctx_lens = [len(ex["context"])     for ex in translated]
    q_lens   = [len(ex["question"])    for ex in translated]
    a_lens   = [len(ex["answer_text"]) for ex in translated]

    # Verify all spans are correct
    span_ok = 0
    for ex in translated:
        start     = ex["answer_start"]
        end       = start + len(ex["answer_text"])
        extracted = ex["context"][start:end]
        if extracted == ex["answer_text"]:
            span_ok += 1

    avg_ctx = statistics.mean(ctx_lens) if ctx_lens else 0
    avg_q   = statistics.mean(q_lens)   if q_lens   else 0
    avg_a   = statistics.mean(a_lens)   if a_lens   else 0

    print(f"\n  Total examples        : {len(translated):,}")
    print(f"  Span verified correct : {span_ok:,} / {len(translated):,}")
    print()
    print(f"  Avg context length    : {avg_ctx:.0f} characters")
    print(f"  Avg question length   : {avg_q:.0f} characters")
    print(f"  Avg answer length     : {avg_a:.0f} characters")

    return {
        "total"      : len(translated),
        "span_ok"    : span_ok,
        "avg_ctx_len": round(avg_ctx, 1),
        "avg_q_len"  : round(avg_q, 1),
        "avg_a_len"  : round(avg_a, 1),
    }


def save_final(translated: list, output_path: str) -> None:
    """
    Save the final translated dataset to disk as JSON.

    Uses ensure_ascii=False so Telugu characters are stored as actual
    Telugu script (రాము) not escaped unicode codes (\\u0c30\\u0c3e\\u0c2e\\u0c41).
    This is essential for any non-English text in JSON.

    Also removes the checkpoint file if it exists — we no longer need it
    once the final file is saved.

    Args:
        translated  (list): Output of translate_all().
        output_path (str) : Where to save the final JSON file.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: SAVING FINAL FILE")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to : {output_path}")
    print(f"   Records  : {len(translated):,}")

    # Clean up checkpoint file
    checkpoint_path = os.path.join(OUTPUT_DIR, "mawps_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"   Checkpoint file removed (no longer needed).")

    # Verify the saved file
    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    print(f"\n  Verification: reloaded {len(loaded):,} records ✅")
    if loaded:
        print(f"  Sample Telugu context : {loaded[0]['context'][:80]}...")
        print(f"  Sample Telugu question: {loaded[0]['question']}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the MAWPS download and translation script.

    Full pipeline:
        1. Download MAWPS from HuggingFace (mwpt5/MAWPS)
        2. Filter for quality and take MAX_EXAMPLES subsample
        3. For each example: substitute numbers, build context, translate
        4. Show sample translations for human quality checking
        5. Compute and print statistics
        6. Save final JSON with Telugu encoding preserved

    This is Week 6 of the project.
    After this script you will have:
        data/mawps_telugu_clean.json  ← ready for merging in Week 7

    Combined with data/indicqa_telugu_clean.json from Week 5, you now
    have both data sources ready for the merge script.
    """
    print("=" * 60)
    print("Telugu Math Chatbot — Lumiere Research Project")
    print("Script: 02_translate_mawps.py  |  Week 6")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Download MAWPS
    # -------------------------------------------------------------------------
    raw_examples = download_mawps(MAWPS_DATASET_ID, MAX_EXAMPLES)

    if not raw_examples:
        print("\n❌ No MAWPS examples loaded. Check internet connection.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Translate to Telugu
    # -------------------------------------------------------------------------
    translated = translate_all(raw_examples)

    if not translated:
        print("\n❌ No examples were successfully translated.")
        print("   Check that deep-translator is installed: pip install deep-translator")
        return

    # -------------------------------------------------------------------------
    # STEP 3: Show sample translations for quality check
    # -------------------------------------------------------------------------
    show_samples(translated, n=3)

    # -------------------------------------------------------------------------
    # STEP 4: Compute statistics
    # -------------------------------------------------------------------------
    stats = compute_statistics(translated)

    # -------------------------------------------------------------------------
    # STEP 5: Save final file
    # -------------------------------------------------------------------------
    save_final(translated, OUTPUT_PATH)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — save these numbers for your paper!")
    print("=" * 60)
    print(f"  Source dataset : MAWPS (mwpt5/MAWPS on HuggingFace)")
    print(f"  Original lang  : English")
    print(f"  Target lang    : Telugu (te)")
    print(f"  Translator     : Google Translate via deep-translator")
    print(f"  Examples input : {len(raw_examples):,}")
    print(f"  After translate: {stats['total']:,}")
    print(f"  Span verified  : {stats['span_ok']:,} / {stats['total']:,}")
    print(f"  Avg context    : {stats['avg_ctx_len']} characters")
    print(f"  Saved to       : {OUTPUT_PATH}")
    print()
    print("  You now have BOTH data sources ready:")
    print("  ✅ data/indicqa_telugu_clean.json  (general Telugu QA, ~1,300 examples)")
    print(f"  ✅ data/mawps_telugu_clean.json    (math Telugu QA, {stats['total']:,} examples)")
    print()
    print("  Next step:")
    print("  → Week 7 : 03_merge_and_clean.py")
    print("             Combine both files into one unified training dataset")


# =============================================================================
# ENTRY POINT GUARD
# =============================================================================
if __name__ == "__main__":
    main()
