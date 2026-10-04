"""
=============================================================================
Script  : 01_download_indicqa.py
Folder  : 01_data/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 5
=============================================================================

PURPOSE
-------
This script downloads the IndicQA Telugu dataset from HuggingFace and
explores its structure so we understand exactly what data we are working
with before we use it to train our model.

It does four things:
    1. Download the IndicQA Telugu JSON file directly from HuggingFace
    2. Explore its structure — fields, sample examples, format
    3. Analyse the data — lengths, quality checks
    4. Save a clean copy to disk for use in later scripts

WHY THIS SCRIPT MATTERS
-----------------------
In research, you must understand your data before you model it.
A model trained on bad or misunderstood data will produce bad results
and you will not know why. This is called "garbage in, garbage out."

HOW WE DOWNLOAD THE DATA (and why it took a few tries!)
--------------------------------------------------------
The IndicQA dataset on HuggingFace has a long history of compatibility
problems — a useful real-world lesson in itself:

  Attempt 1: load_dataset("ai4bharat/IndicQA", "IndicQA.te")
      → FAILED: RuntimeError: Dataset scripts are no longer supported
        (HuggingFace deprecated custom .py loaders in datasets v3+)

  Attempt 2: load_dataset("parquet", data_files=url) using Parquet URLs
      → FAILED: There are no Parquet files — wrong assumption

  Attempt 3: list_repo_files() to discover files dynamically
      → FAILED: 401 Unauthorized (requires a HuggingFace login token)

  SOLUTION (Attempt 4): Browse the actual file tree manually at
      https://huggingface.co/datasets/ai4bharat/IndicQA/tree/main/data
      and discovered the data is stored as plain JSON files:
          data/indicqa.te.json  ← one file, publicly accessible, no auth needed

  We download it directly with Python's requests library — no HuggingFace
  library needed at all. Simple, fast, reliable.

  LESSON FOR YOUR PAPER:
  Real research involves navigating broken tools and changing APIs.
  Document what you tried and why it failed. Reviewers respect this honesty.

WHAT IS IndicQA?
----------------
IndicQA is a dataset by AI4Bharat (IIT Madras) containing question-answering
pairs for 11 Indian languages including Telugu. Each example has:
    "context"      : a Telugu passage (from Wikipedia)
    "question"     : a question about that passage
    "answers"      : the answer text and its character position in the context

The data is factual/general (not math-specific) — which is why we add
math-specific data from MAWPS in Week 6.

WHAT IS SQuAD FORMAT?
---------------------
IndicQA uses SQuAD format — the industry standard for extractive QA.
Each answer is a span extracted directly from the context:
    answer_text  : "అమరావతి"
    answer_start : 42   ← character position in context

INPUTS
------
    No local inputs. Downloads one JSON file (~2MB) from HuggingFace.
    Requires internet connection.

OUTPUTS
-------
    Console : structure, sample examples, statistics
    File    : data/indicqa_telugu_clean.json
              Each record: {id, context, question, answer_text,
                            answer_start, source, split}

HOW TO RUN
----------
    Google Colab : !python 01_download_indicqa.py
    Terminal     : python 01_data/01_download_indicqa.py

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

# requests is the standard Python library for making HTTP requests.
# We use it to download the JSON file directly from HuggingFace.
# This requires no authentication and works regardless of HuggingFace
# library version — much more robust than load_dataset() for this dataset.
import requests

# json is a built-in Python library for reading and writing JSON files.
# The IndicQA data is stored as JSON so we use this to parse it.
import json

# os is a built-in Python library for file system operations.
# We use it to create the output folder and build file paths.
import os

# statistics is a built-in Python library for computing averages.
import statistics

# =============================================================================
# CONSTANTS
# =============================================================================

# Direct URL to the Telugu JSON file in the IndicQA HuggingFace repository.
# Found by browsing: https://huggingface.co/datasets/ai4bharat/IndicQA/tree/main/data
# This URL is public and requires no authentication token.
TELUGU_JSON_URL = (
    "https://huggingface.co/datasets/ai4bharat/IndicQA"
    "/resolve/main/data/indicqa.te.json"
)

# Output folder and file for our cleaned dataset
OUTPUT_DIR  = "data"
OUTPUT_FILE = "indicqa_telugu_clean.json"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# Number of sample examples to print during exploration
NUM_SAMPLES = 3

# =============================================================================
# FUNCTIONS
# =============================================================================

def download_raw_data(url: str) -> dict:
    """
    Download the IndicQA Telugu JSON file directly from HuggingFace.

    WHY requests INSTEAD OF load_dataset()?
    ----------------------------------------
    After three failed attempts using the HuggingFace datasets library
    (deprecated script, wrong file format, auth required), we discovered
    the data is a plain publicly-accessible JSON file. Using requests
    is simpler, faster, and more reliable for this specific dataset.

    The JSON file has this top-level structure:
        {
            "version": 1.0,
            "data": [
                {
                    "title": "...",
                    "paragraphs": [
                        {
                            "context": "Telugu passage...",
                            "qas": [
                                {
                                    "id": "...",
                                    "question": "Telugu question?",
                                    "answers": [
                                        {"text": "answer", "answer_start": 42}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    This is the standard SQuAD JSON format. The nesting is:
        data → paragraphs → qas (question-answer pairs)

    Args:
        url (str): The direct HTTPS URL to the JSON file.

    Returns:
        dict: The parsed JSON content as a Python dictionary.
    """
    print(f"\n⏳ Downloading IndicQA Telugu from HuggingFace...")
    print(f"   URL: {url}")
    print(f"   File size: ~2MB — should be fast\n")

    # requests.get() makes an HTTP GET request to the URL.
    # timeout=60 means: give up if no response within 60 seconds.
    response = requests.get(url, timeout=60)

    # raise_for_status() checks the HTTP status code.
    # If the server returned an error (e.g. 404 Not Found, 401 Unauthorized),
    # this raises an exception with a clear error message.
    # A 200 OK means success.
    response.raise_for_status()

    # response.json() parses the response body as JSON and returns a Python dict.
    data = response.json()

    print(f"✅ Download successful.")
    print(f"   Dataset version : {data.get('version', 'unknown')}")
    print(f"   Top-level keys  : {list(data.keys())}")
    print(f"   Number of topics: {len(data.get('data', []))}")

    return data


def flatten_squad_format(raw_data: dict) -> list:
    """
    Flatten the nested SQuAD JSON structure into a simple list of QA dicts.

    The SQuAD format has three levels of nesting:
        data (topics) → paragraphs → qas (question-answer pairs)

    We flatten this into a simple list where each item is one QA pair
    with its context included directly. This is much easier to work with.

    BEFORE (nested):
        data[0]["paragraphs"][0]["qas"][0] = {
            "id": "te_001",
            "question": "రాముడు ఎక్కడ జన్మించాడు?",
            "answers": [{"text": "అయోధ్య", "answer_start": 15}]
        }
        ... and context is stored at data[0]["paragraphs"][0]["context"]

    AFTER (flat):
        {
            "id"          : "te_001",
            "context"     : "రాముడు అయోధ్య లో జన్మించాడు...",
            "question"    : "రాముడు ఎక్కడ జన్మించాడు?",
            "answer_text" : "అయోధ్య",
            "answer_start": 15,
            "source"      : "indicqa",
            "split"       : "all"   ← IndicQA is not pre-split
        }

    Args:
        raw_data (dict): The parsed JSON dict from download_raw_data().

    Returns:
        list: A flat list of QA pair dicts, one per question.
    """
    print("\n⏳ Flattening nested SQuAD structure...")

    flat_examples = []
    skipped       = 0

    # Loop through topics (e.g. "R.K. Narayan", "Andhra Pradesh history", ...)
    for topic in raw_data.get("data", []):

        # Each topic has multiple paragraphs
        for paragraph in topic.get("paragraphs", []):
            context = paragraph.get("context", "").strip()

            if not context:
                continue  # skip paragraphs with no context

            # Each paragraph has multiple question-answer pairs
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])

                # Skip if no answers provided
                if not answers:
                    skipped += 1
                    continue

                # Take the first answer (there is typically only one)
                answer_text  = answers[0].get("text", "").strip()
                answer_start = answers[0].get("answer_start", -1)

                # Skip empty answers
                if not answer_text:
                    skipped += 1
                    continue

                flat_examples.append({
                    "id"          : qa.get("id", f"te_{len(flat_examples)}"),
                    "context"     : context,
                    "question"    : qa.get("question", "").strip(),
                    "answer_text" : answer_text,
                    "answer_start": answer_start,
                    "source"      : "indicqa",
                    "split"       : "all",  # IndicQA Telugu is not pre-split
                })

    print(f"✅ Flattening complete.")
    print(f"   Total QA pairs extracted : {len(flat_examples):,}")
    print(f"   Skipped (no answer)      : {skipped:,}")

    return flat_examples


def explore_structure(examples: list) -> None:
    """
    Print the structure and sample examples of the flattened dataset.

    A researcher always inspects raw data before processing it.
    This function answers: "What does one row of data look like?"

    We also run a span verification check — extracting
    context[answer_start : answer_start + len(answer_text)]
    and confirming it matches answer_text. This is a quick data
    quality check that would catch encoding issues or off-by-one errors.

    Args:
        examples (list): Output of flatten_squad_format().

    Returns:
        None. Prints to console.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: DATASET STRUCTURE")
    print("=" * 60)

    print(f"\nFields in each record: {list(examples[0].keys())}")
    print(f"\n--- Sample examples ({NUM_SAMPLES} shown) ---\n")

    for i in range(min(NUM_SAMPLES, len(examples))):
        ex = examples[i]

        print(f"Example {i + 1}:")
        print(f"  ID           : {ex['id']}")
        # Truncate long contexts for readability — full context is preserved in the data
        ctx_preview = ex["context"][:120] + "..." if len(ex["context"]) > 120 else ex["context"]
        print(f"  Context      : {ctx_preview}")
        print(f"  Question     : {ex['question']}")
        print(f"  Answer text  : {ex['answer_text']}")
        print(f"  Answer start : {ex['answer_start']}  ← character index in context")

        # Span verification: extract the substring at answer_start
        # and check it matches answer_text
        start = ex["answer_start"]
        end   = start + len(ex["answer_text"])
        if start >= 0:
            extracted = ex["context"][start:end]
            if extracted == ex["answer_text"]:
                print(f"  Span check   : ✅ context[{start}:{end}] matches answer")
            else:
                print(f"  Span check   : ❌ MISMATCH — extracted '{extracted}'")
        print()


def analyse_statistics(examples: list) -> dict:
    """
    Compute and print summary statistics about the dataset.

    Why this matters:
        1. Catches data quality issues before they cause training bugs
        2. These numbers go into the Data section of your research paper

    Stats computed:
        - Total examples
        - Average context / question / answer lengths in characters
        - Count of span mismatches (data quality issues)
        - Count of contexts that may exceed MuRIL's token limit

    Args:
        examples (list): Output of flatten_squad_format().

    Returns:
        dict: Summary stats for use in the paper.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: DATA STATISTICS")
    print("=" * 60)

    ctx_lens    = [len(ex["context"])     for ex in examples]
    q_lens      = [len(ex["question"])    for ex in examples]
    a_lens      = [len(ex["answer_text"]) for ex in examples]

    # Count span mismatches — these are data quality issues
    mismatches = 0
    for ex in examples:
        start     = ex["answer_start"]
        end       = start + len(ex["answer_text"])
        extracted = ex["context"][start:end]
        if extracted != ex["answer_text"]:
            mismatches += 1

    avg_ctx = statistics.mean(ctx_lens)
    avg_q   = statistics.mean(q_lens)
    avg_a   = statistics.mean(a_lens)

    print(f"\n  Total examples        : {len(examples):,}")
    print(f"  Span mismatches       : {mismatches:,}  ← will be filtered in cleaning")
    print()
    print(f"  Avg context length    : {avg_ctx:.0f} characters")
    print(f"  Avg question length   : {avg_q:.0f} characters")
    print(f"  Avg answer length     : {avg_a:.0f} characters")

    # MuRIL processes max 384 tokens. Telugu characters are typically
    # 1–3 tokens each. Contexts over ~250 characters risk truncation,
    # which can cut off the answer and make the example unlearnable.
    THRESHOLD   = 250
    long_count  = sum(1 for l in ctx_lens if l > THRESHOLD)
    long_pct    = long_count / len(examples) * 100

    print()
    print(f"  Contexts > {THRESHOLD} chars     : {long_count:,} ({long_pct:.1f}%)")
    print(f"  ⚠️  Long contexts are auto-truncated by the tokenizer.")
    print(f"     Answers in the truncated portion cannot be learned.")
    print(f"     Document this as a limitation in the paper.")

    return {
        "total"       : len(examples),
        "avg_ctx_len" : round(avg_ctx, 1),
        "avg_q_len"   : round(avg_q, 1),
        "avg_a_len"   : round(avg_a, 1),
        "mismatches"  : mismatches,
    }


def clean_and_save(examples: list, output_path: str) -> list:
    """
    Remove bad examples and save the clean dataset to disk as JSON.

    FILTERS applied:
        1. Remove examples where answer_start < 0 (unknown position)
        2. Remove examples where the span is inconsistent:
           context[answer_start : answer_start + len(answer)] != answer
        3. Remove examples with empty question or answer text

    SAVING:
        JSON format with ensure_ascii=False — CRITICAL for Telugu.
        Without this, Telugu characters are saved as \\u0c30\\u0c3e...
        instead of actual Telugu script రా. Always use ensure_ascii=False
        when saving any non-English text.

    Args:
        examples    (list): Output of flatten_squad_format().
        output_path (str) : Where to save the clean JSON file.

    Returns:
        list: The cleaned examples that were saved.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: CLEANING AND SAVING")
    print("=" * 60)

    cleaned           = []
    removed_no_start  = 0
    removed_mismatch  = 0
    removed_empty     = 0

    for ex in examples:

        # Filter 1: must have a valid start position
        if ex["answer_start"] < 0:
            removed_no_start += 1
            continue

        # Filter 2: span must be consistent with context
        start     = ex["answer_start"]
        end       = start + len(ex["answer_text"])
        extracted = ex["context"][start:end]
        if extracted != ex["answer_text"]:
            removed_mismatch += 1
            continue

        # Filter 3: must have non-empty question and answer
        if not ex["question"].strip() or not ex["answer_text"].strip():
            removed_empty += 1
            continue

        cleaned.append(ex)

    total_removed = removed_no_start + removed_mismatch + removed_empty
    print(f"\n  Filtering results:")
    print(f"    Removed (bad start index) : {removed_no_start:,}")
    print(f"    Removed (span mismatch)   : {removed_mismatch:,}")
    print(f"    Removed (empty text)      : {removed_empty:,}")
    print(f"    Total removed             : {total_removed:,}")
    print(f"    Examples kept             : {len(cleaned):,}")

    # Create output folder if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save to JSON
    # indent=2 makes the file human-readable
    # ensure_ascii=False preserves Telugu characters as actual script
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to : {output_path}")
    print(f"   Records  : {len(cleaned):,}")

    return cleaned


def verify_saved_file(output_path: str) -> None:
    """
    Reload the saved JSON file and confirm it was written correctly.

    Checks:
        1. File is valid JSON and loads without errors
        2. Telugu text is preserved as actual script (not \\u codes)
        3. All expected fields are present

    Always verify files after writing — a simple habit that catches
    encoding bugs before they cause problems in downstream scripts.

    Args:
        output_path (str): Path to the saved JSON file.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: FILE VERIFICATION")
    print("=" * 60)

    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    print(f"\n  File loaded : ✅ valid JSON")
    print(f"  Records     : {len(loaded):,}")

    if loaded:
        print(f"\n  First record:")
        for key, value in loaded[0].items():
            display = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"    {key:15s}: {display}")

        print(f"\n  Telugu rendering check:")
        print(f"    Question : {loaded[0]['question']}")
        print(f"    Answer   : {loaded[0]['answer_text']}")
        print()
        print(f"  Telugu script visible → encoding is correct ✅")
        print(f"  \\u0c30... codes visible → re-run with ensure_ascii=False ❌")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the IndicQA Telugu dataset download script.

    Full pipeline:
        1. Download indicqa.te.json directly via requests (no auth needed)
        2. Flatten the nested SQuAD structure into a simple list
        3. Explore structure — fields, sample examples, span checks
        4. Compute statistics — lengths, quality flags
        5. Clean — filter bad examples
        6. Save as JSON with correct Telugu (UTF-8) encoding
        7. Verify the saved file is readable and correct

    This is Week 5. After this script you will have:
        data/indicqa_telugu_clean.json  ← ready for merging in Week 7
    """
    print("=" * 60)
    print("Telugu Math Chatbot — Lumiere Research Project")
    print("Script: 01_download_indicqa.py  |  Week 5")
    print("=" * 60)

    # STEP 1: Download the raw JSON file
    raw_data = download_raw_data(TELUGU_JSON_URL)

    # STEP 2: Flatten nested SQuAD structure into simple list
    examples = flatten_squad_format(raw_data)

    if not examples:
        print("\n❌ No examples extracted. Check the URL and JSON structure.")
        return

    # STEP 3: Explore structure
    explore_structure(examples)

    # STEP 4: Compute statistics
    stats = analyse_statistics(examples)

    # STEP 5 & 6: Clean and save
    cleaned = clean_and_save(examples, OUTPUT_PATH)

    # STEP 7: Verify
    verify_saved_file(OUTPUT_PATH)

    # FINAL SUMMARY — note these for your paper
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — save these numbers for your paper!")
    print("=" * 60)
    print(f"  Dataset        : AI4Bharat IndicQA (Telugu)")
    print(f"  Download method: Direct JSON via requests (no auth required)")
    print(f"  URL            : {TELUGU_JSON_URL}")
    print(f"  Raw examples   : {stats['total']:,}")
    print(f"  After cleaning : {len(cleaned):,}")
    print(f"  Avg context    : {stats['avg_ctx_len']} characters")
    print(f"  Avg question   : {stats['avg_q_len']} characters")
    print(f"  Avg answer     : {stats['avg_a_len']} characters")
    print(f"  Saved to       : {OUTPUT_PATH}")
    print()
    print("  Next steps:")
    print("  → Week 6 : 02_translate_mawps.py — download math problems,")
    print("              translate English → Telugu using deep-translator")
    print("  → Week 7 : 03_merge_and_clean.py — combine both sources")
    print("              into one unified training dataset")


# =============================================================================
# ENTRY POINT GUARD
# =============================================================================
if __name__ == "__main__":
    main()
