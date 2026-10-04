"""
=============================================================================
Script  : 01_load_muril_baseline.py
Folder  : 03_model/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 4
=============================================================================

PURPOSE
-------
This is your very first hands-on script. Its job is simple:
    1. Download the MuRIL pretrained model from HuggingFace
    2. Load its tokenizer (the part that converts text to numbers)
    3. Run a few sample Telugu math questions through the model
    4. Print the results so you can see it working

No training happens here. This script is purely about getting comfortable
with the HuggingFace library and proving that MuRIL can process Telugu text.

WHY THIS SCRIPT MATTERS
-----------------------
Before you can fine-tune a model, you need to understand the tool you are
working with. This script teaches you the THREE building blocks you will
reuse in every script that follows:

    Tokenizer  →  converts raw Telugu text into token IDs (numbers)
    Model      →  the neural network that processes those numbers
    Pipeline   →  a convenience wrapper that combines tokenizer + model

Think of it like this:
    Raw text  →  [Tokenizer]  →  Numbers  →  [Model]  →  Answer

WHAT IS MuRIL?
--------------
MuRIL stands for Multilingual Representations for Indian Languages.
It was built by Google and pretrained on text from 17 Indian languages
including Telugu, Hindi, Tamil, Kannada, and others.

It uses the BERT architecture — a transformer model that reads text
bidirectionally (it looks at words before AND after each word to understand
context). This makes it much better than older models at understanding meaning.

HuggingFace model card: https://huggingface.co/google/muril-base-cased

WHAT IS A TOKENIZER?
--------------------
Computers cannot read text directly — they can only process numbers.
A tokenizer converts each word (or part of a word) into a unique number
called a token ID.

Example (simplified):
    "రాము"   →  [2847]
    "ఆపిల్లు" →  [5621, 3302]   ← some words split into sub-word pieces

MuRIL uses a tokenizer called SentencePiece which handles Telugu script well.

WHAT IS THE TASK FORMAT?
------------------------
This script uses Question Answering (QA) in extractive format:
    - You provide a CONTEXT: a short passage containing the answer
    - You provide a QUESTION: what you want to know
    - The model EXTRACTS a span of text from the context as its answer

This is the same format as the Stanford Question Answering Dataset (SQuAD),
which is the standard benchmark for this type of task.

INPUTS
------
    No file inputs. The model is downloaded automatically from HuggingFace.
    Requires internet connection on first run (~900MB download).
    The model is cached locally after the first download.

OUTPUTS
-------
    Printed results to the console showing:
        - Tokenizer vocabulary size
        - How sample Telugu sentences are tokenized
        - Model answers to sample Telugu math questions

HOW TO RUN
----------
    Option A — Google Colab (recommended):
        Upload this file to Colab, then run:
            !python 01_load_muril_baseline.py

    Option B — Local terminal:
        python 03_model/01_load_muril_baseline.py

EXPECTED OUTPUT (approximate)
------------------------------
    Tokenizer loaded. Vocabulary size: 197,285
    Model loaded. Parameters: 236,491,776

    --- Tokenization Example ---
    Input text  : రాము దగ్గర పది ఆపిల్లు ఉన్నాయి
    Token IDs   : [3, 29847, 46231, 12045, 87342, 23901, 4]
    Tokens      : ['[CLS]', 'రాము', 'దగ్గర', 'పది', 'ఆపిల్', '##లు', '[SEP]']
    Token count : 7

    --- Question Answering Example 1 ---
    Context  : రాము దగ్గర పది ఆపిల్లు ఉన్నాయి. అతను మూడు ఆపిల్లు తన స్నేహితుడికి ఇచ్చాడు.
    Question : రాము దగ్గర ఇప్పుడు ఎన్ని ఆపిల్లు ఉన్నాయి?
    Answer   : మూడు   (NOTE: baseline without fine-tuning — answer may be wrong!)
    Score    : 0.043

    [... more examples ...]

    ⚠️  NOTICE: The baseline model answers may be WRONG or nonsensical.
    This is expected! MuRIL was not trained on math problems.
    Fine-tuning (Week 8) will teach it to answer correctly.

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

# transformers is the HuggingFace library — the most important import in this project.
# We import:
#   AutoTokenizer                 : automatically loads the right tokenizer for any model
#   AutoModelForQuestionAnswering : loads a model set up for the QA task
#
# NOTE ON PIPELINE COMPATIBILITY:
#   transformers 5.x restructured pipelines significantly — "question-answering"
#   was removed as a task string and QuestionAnsweringPipeline was moved/renamed.
#   Rather than chase those moving parts, we do inference MANUALLY using PyTorch
#   directly. This is actually better for learning because it shows exactly what
#   happens inside the pipeline: tokenize → forward pass → decode answer span.
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# torch is PyTorch — the deep learning framework that runs underneath HuggingFace.
# We use it for:
#   torch.no_grad()       : disables gradient tracking during inference (saves memory)
#   torch.cuda.is_available() : checks if a GPU is available
#   torch.softmax()       : converts raw model scores into probabilities
import torch

# json is a built-in Python library for reading and writing JSON files.
# JSON (JavaScript Object Notation) is the format we store our dataset in.
import json

# =============================================================================
# CONSTANTS
# All configuration values go here at the top — never bury them in functions.
# This makes it easy to change settings without hunting through the code.
# =============================================================================

# The HuggingFace model identifier for MuRIL.
# This string tells HuggingFace exactly which model to download.
# You can browse models at https://huggingface.co/models
MODEL_NAME = "google/muril-base-cased"

# Maximum number of tokens the model can process in one pass.
# MuRIL supports up to 512. We use 384 to leave room for both
# the question and the context without exceeding the limit.
MAX_TOKEN_LENGTH = 384

# =============================================================================
# SAMPLE DATA
# A small set of Telugu math Q&A examples for testing the baseline.
# Each example has three parts:
#   "context"  — a short passage that contains the answer
#   "question" — what we are asking
#   "answer"   — the correct answer (so we can see if the model gets it right)
#
# NOTE: These are written in Telugu script. If they display as boxes or
# question marks in your terminal, that is a font issue — the data is correct.
# Google Colab displays Telugu script properly.
# =============================================================================

SAMPLE_QA_PAIRS = [
    {
        "context": (
            "రాము దగ్గర పది ఆపిల్లు ఉన్నాయి. "
            "అతను మూడు ఆపిల్లు తన స్నేహితుడికి ఇచ్చాడు. "
            "ఇప్పుడు రాము దగ్గర ఏడు ఆపిల్లు మాత్రమే ఉన్నాయి."
        ),
        # Translation: Ramu had ten apples. He gave three apples to his friend.
        #              Now Ramu has only seven apples.
        "question": "రాము దగ్గర ఇప్పుడు ఎన్ని ఆపిల్లు ఉన్నాయి?",
        # Translation: How many apples does Ramu have now?
        "answer": "ఏడు"
        # Translation: seven
    },
    {
        "context": (
            "సీత దగ్గర ఇరవై రూపాయలు ఉన్నాయి. "
            "ఆమె పుస్తకం కోసం పన్నెండు రూపాయలు ఖర్చు చేసింది. "
            "ఆమె దగ్గర ఇప్పుడు ఎనిమిది రూపాయలు మిగిలాయి."
        ),
        # Translation: Sita had twenty rupees. She spent twelve rupees on a book.
        #              She now has eight rupees left.
        "question": "సీత దగ్గర ఎంత డబ్బు మిగిలింది?",
        # Translation: How much money does Sita have left?
        "answer": "ఎనిమిది రూపాయలు"
        # Translation: eight rupees
    },
    {
        "context": (
            "ఒక తరగతిలో ముప్పై విద్యార్థులు ఉన్నారు. "
            "వారిలో పదిహేను మంది అమ్మాయిలు మరియు పదిహేను మంది అబ్బాయిలు ఉన్నారు."
        ),
        # Translation: There are thirty students in a class.
        #              Among them, fifteen are girls and fifteen are boys.
        "question": "తరగతిలో ఎంత మంది అమ్మాయిలు ఉన్నారు?",
        # Translation: How many girls are in the class?
        "answer": "పదిహేను"
        # Translation: fifteen
    },
    {
        "context": (
            "అనిల్ ప్రతి రోజు ఐదు కిలోమీటర్లు నడుస్తాడు. "
            "ఒక వారంలో అతను మొత్తం ముప్పై అయిదు కిలోమీటర్లు నడుస్తాడు."
        ),
        # Translation: Anil walks five kilometres every day.
        #              In one week he walks thirty-five kilometres in total.
        "question": "అనిల్ వారంలో మొత్తం ఎంత దూరం నడుస్తాడు?",
        # Translation: How far does Anil walk in a week in total?
        "answer": "ముప్పై అయిదు కిలోమీటర్లు"
        # Translation: thirty-five kilometres
    },
]

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Download and load the MuRIL tokenizer from HuggingFace.

    The tokenizer is responsible for converting raw text into token IDs
    (numbers) that the model can understand. Every model has its own
    tokenizer that must match exactly.

    On the first run, this downloads the tokenizer files (~1MB) from
    HuggingFace and caches them locally. Subsequent runs use the cache.

    Args:
        model_name (str): The HuggingFace model identifier string.
                          For MuRIL this is "google/muril-base-cased".

    Returns:
        AutoTokenizer: The loaded tokenizer object, ready to use.

    Example:
        >>> tokenizer = load_tokenizer("google/muril-base-cased")
        >>> print(tokenizer.vocab_size)
        197285
    """
    print(f"\n⏳ Loading tokenizer for: {model_name}")
    print("   (First run downloads ~1MB — subsequent runs use local cache)")

    # AutoTokenizer.from_pretrained automatically detects the right tokenizer
    # class for the model and downloads it. The string must match the model card
    # URL on HuggingFace: https://huggingface.co/google/muril-base-cased
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"✅ Tokenizer loaded successfully.")
    print(f"   Vocabulary size : {tokenizer.vocab_size:,} tokens")
    print(f"   Tokenizer type  : {type(tokenizer).__name__}")

    return tokenizer


def load_model(model_name: str) -> AutoModelForQuestionAnswering:
    """
    Download and load the MuRIL model set up for Question Answering.

    This loads MuRIL with a QA "head" on top — a small extra layer that
    outputs two numbers per token: one score for "does the answer START
    here?" and one for "does the answer END here?". The model predicts
    the span of text in the context that answers the question.

    On the first run, this downloads the model weights (~900MB) from
    HuggingFace. This may take a few minutes on a slow connection.
    Subsequent runs use the local cache (~/.cache/huggingface/).

    Args:
        model_name (str): The HuggingFace model identifier string.

    Returns:
        AutoModelForQuestionAnswering: The loaded model, ready for inference.

    Example:
        >>> model = load_model("google/muril-base-cased")
        >>> total_params = sum(p.numel() for p in model.parameters())
        >>> print(f"Parameters: {total_params:,}")
        Parameters: 236,491,776
    """
    print(f"\n⏳ Loading model: {model_name}")
    print("   (First run downloads ~900MB — this may take a few minutes)")

    # AutoModelForQuestionAnswering loads the model with a QA-specific output layer
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Count the total number of trainable parameters.
    # Parameters are the numerical weights inside the model that were learned
    # during pretraining. More parameters = more capacity to learn, but also
    # more memory and compute needed.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model loaded successfully.")
    print(f"   Total parameters     : {total_params:,}")
    print(f"   Trainable parameters : {trainable_params:,}")
    print(f"   Model architecture   : {type(model).__name__}")

    return model


def check_device() -> str:
    """
    Check whether a GPU is available and return the appropriate device string.

    Deep learning is much faster on a GPU (Graphics Processing Unit) than
    on a CPU. Google Colab provides a free GPU — make sure to enable it in
    Runtime → Change runtime type → T4 GPU.

    Returns:
        str: Either "cuda" (GPU available) or "cpu" (no GPU found).

    Example:
        >>> device = check_device()
        🖥️  Device: CPU (no GPU found — training will be slow)
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"\n🚀 Device: GPU detected — {device_name}")
        print("   Training and inference will be fast.")
        return "cuda"
    else:
        print("\n🖥️  Device: CPU (no GPU found)")
        print("   For this baseline script, CPU is fine.")
        print("   For fine-tuning (Week 8), enable GPU in Colab:")
        print("   Runtime → Change runtime type → T4 GPU")
        return "cpu"


def explore_tokenizer(tokenizer: AutoTokenizer, sample_texts: list) -> None:
    """
    Show how the tokenizer breaks Telugu text into tokens.

    This function is purely educational — it prints a detailed breakdown
    of how the tokenizer processes text so you can see what the model
    actually "sees" when it reads Telugu.

    Key concept: Tokenizers often split words into smaller pieces called
    sub-words. For example, "ఆపిల్లు" might become ["ఆపిల్", "##లు"].
    The "##" prefix means "this piece continues the previous token".
    This is how the tokenizer handles words it has not seen before.

    Args:
        tokenizer (AutoTokenizer): The loaded MuRIL tokenizer.
        sample_texts (list of str): A list of Telugu sentences to tokenize.

    Returns:
        None. Prints results to the console.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: TOKENIZATION EXPLORATION")
    print("=" * 60)
    print("This shows how MuRIL converts Telugu text into numbers.\n")

    for i, text in enumerate(sample_texts, start=1):

        # tokenizer(text) returns a BatchEncoding object with:
        #   input_ids      : list of token ID numbers
        #   attention_mask : list of 1s (real tokens) and 0s (padding)
        #   token_type_ids : 0 for first sentence, 1 for second (used in QA)
        encoding = tokenizer(text, return_tensors="pt")

        # input_ids is a 2D tensor (batch_size=1, sequence_length)
        # .tolist()[0] converts it to a flat Python list of integers
        token_ids = encoding["input_ids"].tolist()[0]

        # convert_ids_to_tokens turns token ID numbers back into readable strings
        # This lets us see exactly how the text was split
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        print(f"Example {i}:")
        print(f"  Input text  : {text}")
        print(f"  Token count : {len(token_ids)} tokens")
        print(f"  Token IDs   : {token_ids}")
        print(f"  Tokens      : {tokens}")
        print()


def predict_answer(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    device: str
) -> dict:
    """
    Run a single question-answering inference pass manually using PyTorch.

    This function does exactly what a HuggingFace pipeline does internally,
    but written out step by step so you can see every stage:

        Step 1 — Tokenize  : convert question + context text into token IDs
        Step 2 — Forward   : pass token IDs through the model to get logit scores
        Step 3 — Argmax    : find which token positions have the highest
                             start-score and end-score
        Step 4 — Decode    : convert those token positions back into text

    WHY MANUAL INFERENCE INSTEAD OF pipeline()?
    --------------------------------------------
    transformers 5.x restructured its pipeline internals and moved/renamed
    QuestionAnsweringPipeline. Rather than depend on those moving parts,
    we call the model directly with PyTorch. This approach:
        - Works with ALL versions of transformers
        - Shows you exactly what is happening (no magic)
        - Is the same pattern used in research code worldwide

    HOW EXTRACTIVE QA WORKS:
    -------------------------
    The model outputs two vectors, each of length = number of tokens:
        start_logits : a score for each token being the START of the answer
        end_logits   : a score for each token being the END of the answer

    We pick the token with the highest start score and the token with the
    highest end score, then extract all tokens between them as the answer.

    Args:
        model       : The loaded MuRIL QA model.
        tokenizer   : The loaded MuRIL tokenizer.
        question (str) : The Telugu question to ask.
        context  (str) : The Telugu passage that contains the answer.
        device   (str) : "cuda" for GPU, "cpu" for CPU.

    Returns:
        dict with keys:
            "answer" (str)   — the extracted answer text
            "score"  (float) — combined confidence score (0.0 to 1.0)
            "start"  (int)   — token index where answer starts
            "end"    (int)   — token index where answer ends
    """
    # -------------------------------------------------------------------------
    # STEP 1: TOKENIZE
    # We pass BOTH the question and context together to the tokenizer.
    # The tokenizer produces a single sequence:
    #   [CLS] question tokens [SEP] context tokens [SEP]
    # token_type_ids marks which tokens belong to Q (0) vs context (1).
    # -------------------------------------------------------------------------
    inputs = tokenizer(
        question,                  # First argument = question
        context,                   # Second argument = context
        return_tensors="pt",       # "pt" = return PyTorch tensors (not lists)
        truncation=True,           # Cut off if too long for the model
        max_length=MAX_TOKEN_LENGTH,
        padding=True,              # Pad to a consistent length
    )

    # Move the input tensors to the same device as the model (GPU or CPU).
    # .to(device) is PyTorch syntax for moving data between devices.
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # -------------------------------------------------------------------------
    # STEP 2: FORWARD PASS
    # torch.no_grad() tells PyTorch NOT to compute gradients.
    # Gradients are only needed during training (they help adjust model weights).
    # During inference we skip them to save memory and run faster.
    # -------------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.start_logits : shape (1, sequence_length) — score per token for START
        # outputs.end_logits   : shape (1, sequence_length) — score per token for END

    # -------------------------------------------------------------------------
    # STEP 3: FIND BEST START AND END TOKEN POSITIONS
    # .argmax() returns the index of the maximum value in the tensor.
    # .item() converts a single-element tensor to a plain Python integer.
    # -------------------------------------------------------------------------
    start_index = outputs.start_logits.argmax(dim=-1).item()
    end_index   = outputs.end_logits.argmax(dim=-1).item()

    # Safety check: end must be >= start.
    # If the model predicts end before start (a common failure mode on untrained
    # models), we set end = start so we return at least one token.
    if end_index < start_index:
        end_index = start_index

    # -------------------------------------------------------------------------
    # STEP 4: COMPUTE CONFIDENCE SCORE
    # Convert raw logits to probabilities using softmax, then multiply the
    # start and end probabilities together as a combined confidence score.
    # -------------------------------------------------------------------------
    start_probs = torch.softmax(outputs.start_logits, dim=-1)
    end_probs   = torch.softmax(outputs.end_logits,   dim=-1)
    score = (start_probs[0, start_index] * end_probs[0, end_index]).item()

    # -------------------------------------------------------------------------
    # STEP 5: DECODE ANSWER
    # Convert the winning token IDs back into a readable text string.
    # inputs["input_ids"][0] is the flat list of token IDs for this example.
    # [start_index : end_index + 1] slices out the answer span.
    # tokenizer.decode() converts IDs back to text, skipping special tokens
    # like [CLS] and [SEP].
    # -------------------------------------------------------------------------
    answer_ids = inputs["input_ids"][0][start_index : end_index + 1]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return {
        "answer": answer_text,
        "score" : round(score, 4),
        "start" : start_index,
        "end"   : end_index,
    }


def run_baseline_qa(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    qa_pairs: list,
    device: str
) -> list:
    """
    Run the zero-shot baseline model on a list of Telugu math Q&A pairs.

    "Zero-shot" means the model has NOT been trained on math problems —
    we are testing MuRIL straight out of the box to see what it can do.
    The answers will likely be wrong or partially correct. That is expected!

    This gives us a baseline score to compare against after fine-tuning
    in Week 9. A good experiment always compares before vs. after.

    Args:
        model     : The loaded MuRIL QA model.
        tokenizer : The loaded MuRIL tokenizer.
        qa_pairs (list of dict): Each dict must have keys:
                                    "context"  (str) — passage with the answer
                                    "question" (str) — the question to answer
                                    "answer"   (str) — the correct answer
        device (str): "cuda" or "cpu".

    Returns:
        list of dict: The input list with added "prediction" and "is_correct"
                      keys for each item.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: BASELINE QUESTION ANSWERING")
    print("=" * 60)
    print("Testing MuRIL BEFORE fine-tuning. Answers may be wrong — that is ok!\n")

    results = []

    for i, pair in enumerate(qa_pairs, start=1):
        print(f"--- Example {i} ---")
        print(f"Context  : {pair['context']}")
        print(f"Question : {pair['question']}")
        print(f"Correct  : {pair['answer']}")

        # Call our manual inference function
        prediction = predict_answer(
            model=model,
            tokenizer=tokenizer,
            question=pair["question"],
            context=pair["context"],
            device=device,
        )

        print(f"Predicted: {prediction['answer']}")
        print(f"Score    : {prediction['score']:.3f}  (confidence, 0=low, 1=high)")

        # Simple check: did the model get it right?
        # We use 'in' rather than == because the model might predict a sub-span.
        is_correct = (
            pair["answer"] in prediction["answer"] or
            prediction["answer"] in pair["answer"]
        )
        print(f"Correct? : {'✅ YES' if is_correct else '❌ NO (expected before fine-tuning)'}")
        print()

        results.append({
            **pair,
            "prediction": prediction,
            "is_correct": is_correct,
        })

    return results


def print_summary(results: list) -> None:
    """
    Print a summary of baseline performance across all test examples.

    This gives a quick overview of how many questions the model answered
    correctly before any fine-tuning. We will compare this to the fine-tuned
    model's performance in Week 9 to show improvement.

    Args:
        results (list of dict): Output from run_baseline_qa(), each containing
                                an "is_correct" boolean key.

    Returns:
        None. Prints summary to console.
    """
    print("=" * 60)
    print("SUMMARY — BASELINE PERFORMANCE (before fine-tuning)")
    print("=" * 60)

    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Total questions : {total}")
    print(f"Correct answers : {correct}")
    print(f"Accuracy        : {accuracy:.1f}%")
    print()
    print("⚠️  IMPORTANT NOTE:")
    print("   A low accuracy here is completely expected and is NOT a problem.")
    print("   MuRIL was pretrained on general Telugu text — not math problems.")
    print("   After fine-tuning in Week 8–9, accuracy will improve significantly.")
    print("   Save these baseline numbers — you will cite them in your paper!")
    print()
    print("   Next steps:")
    print("   → Week 5: Download and explore the IndicQA Telugu dataset")
    print("   → Week 6: Build the Telugu math Q&A dataset")
    print("   → Week 7: Clean data and explore tokenization")
    print("   → Week 8: Fine-tune MuRIL and see the improvement!")


# =============================================================================
# MAIN FUNCTION
# The main() function is the entry point — it calls all other functions
# in the right order. This is a standard pattern in Python scripts.
# =============================================================================

def main():
    """
    Main entry point for the MuRIL baseline loading and testing script.

    Runs the full pipeline:
        1. Check available hardware (CPU vs GPU)
        2. Load the MuRIL tokenizer
        3. Load the MuRIL model for Question Answering
        4. Explore how the tokenizer processes Telugu text
        5. Run baseline QA using manual PyTorch inference (no pipeline wrapper)
        6. Print a performance summary

    This is Week 4 of the project — your first hands-on coding session.
    No training happens here. This is purely about loading the model and
    understanding the HuggingFace + PyTorch tools you will use throughout.

    NOTE: We use manual inference (predict_answer) instead of HuggingFace
    pipeline() because transformers 5.x restructured its pipeline internals.
    Manual inference works with all versions and teaches you more.
    """
    print("=" * 60)
    print("Telugu Math Chatbot — Lumiere Research Project")
    print("Script: 01_load_muril_baseline.py  |  Week 4")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Check hardware
    # -------------------------------------------------------------------------
    device = check_device()

    # -------------------------------------------------------------------------
    # STEP 2: Load tokenizer
    # -------------------------------------------------------------------------
    tokenizer = load_tokenizer(MODEL_NAME)

    # -------------------------------------------------------------------------
    # STEP 3: Load model
    # -------------------------------------------------------------------------
    model = load_model(MODEL_NAME)

    # -------------------------------------------------------------------------
    # STEP 4: Explore tokenization
    # Show how MuRIL breaks Telugu sentences into tokens
    # -------------------------------------------------------------------------
    sample_sentences = [
        "రాము దగ్గర పది ఆపిల్లు ఉన్నాయి",          # Ramu has ten apples
        "ఒక తరగతిలో ముప్పై విద్యార్థులు ఉన్నారు",   # There are thirty students in a class
        "ఐదు కి మూడు కలిపితే ఎంత అవుతుంది?",        # What is five plus three?
    ]
    explore_tokenizer(tokenizer, sample_sentences)

    # -------------------------------------------------------------------------
    # STEP 5: Run baseline inference on sample Telugu math questions
    #
    # We pass the model, tokenizer, and device directly to run_baseline_qa.
    # Internally it calls predict_answer() for each example, which does a
    # full manual PyTorch forward pass — no pipeline() wrapper needed.
    # This approach works with all versions of transformers.
    # -------------------------------------------------------------------------
    results = run_baseline_qa(
        model=model,
        tokenizer=tokenizer,
        qa_pairs=SAMPLE_QA_PAIRS,
        device=device,
    )

    # -------------------------------------------------------------------------
    # STEP 6: Print summary
    # -------------------------------------------------------------------------
    print_summary(results)


# =============================================================================
# ENTRY POINT GUARD
#
# This block ensures that main() is only called when this script is run
# directly (e.g. "python 01_load_muril_baseline.py"), NOT when it is
# imported by another script.
#
# This is a standard Python best practice that you will see in every
# well-written Python codebase.
# =============================================================================
if __name__ == "__main__":
    main()
