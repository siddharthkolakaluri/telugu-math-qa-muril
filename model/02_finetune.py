"""
=============================================================================
Script  : 02_finetune.py
Folder  : 03_model/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 8
=============================================================================

PURPOSE
-------
This script fine-tunes MuRIL on our Telugu math QA dataset.
Fine-tuning means taking a pretrained model that already understands
Telugu and teaching it specifically to answer math questions.

After this script runs you will have a trained model saved to disk
that can answer Telugu elementary math questions — ready for evaluation
in Week 9 and the Gradio chatbot demo in Week 10.

WHAT HAPPENS DURING FINE-TUNING?
---------------------------------
Fine-tuning is a form of supervised learning. For each training example:

    1. FORWARD PASS
       The model reads the context + question and predicts:
           - Which token is most likely the START of the answer
           - Which token is most likely the END of the answer

    2. LOSS COMPUTATION
       We compare the model's predictions to the correct answer positions.
       The difference is the LOSS — a number measuring how wrong the
       model was. High loss = very wrong. Low loss = nearly correct.

    3. BACKWARD PASS (backpropagation)
       PyTorch computes the gradient — the direction to adjust each of
       the 236 million model weights to reduce the loss slightly.

    4. WEIGHT UPDATE (optimizer step)
       The optimizer (AdamW) nudges each weight by a tiny amount in
       the direction that reduces loss. The size of the nudge is the
       LEARNING RATE.

    5. REPEAT
       Steps 1–4 happen for every example in every epoch.
       An epoch is one full pass through the training data.
       We train for NUM_EPOCHS epochs.

    Over thousands of these updates, the model gradually learns to
    predict the correct answer spans for Telugu math questions.

KEY CONCEPTS
------------
    Learning Rate : how big each weight update step is.
                    Too high → model overshoots and never converges.
                    Too low  → model learns very slowly.
                    We use 2e-5 (0.00002) — standard for BERT fine-tuning.

    Batch Size    : how many examples to process before updating weights.
                    Larger = more stable gradients but more GPU memory.
                    We use 8 — safe for Colab T4 GPU.

    Epoch         : one full pass through all training examples.
                    We train for 3 epochs — standard for small datasets.

    Overfitting   : when the model memorises training data instead of
                    learning general patterns. Signs: training loss keeps
                    falling but validation loss stops falling or rises.
                    We monitor this using eval_strategy="epoch".

    AdamW         : the optimizer — the algorithm that updates weights.
                    AdamW is Adam with Weight Decay, the standard choice
                    for transformer fine-tuning.

DATASET FORMAT
--------------
The training data (from Week 7) is in flat JSON format:
    {
        "id"          : "mawps_0001",
        "context"     : "Telugu math passage...",
        "question"    : "సమాధానం ఏమిటి?",
        "answer_text" : "7",
        "answer_start": 85,
        "source"      : "mawps_translated"
    }

We convert this to the format HuggingFace Trainer expects:
    input_ids, attention_mask, token_type_ids (from tokenizer)
    start_positions, end_positions (token indices of the answer)

Note: answer_start is a CHARACTER position. We must convert it to
a TOKEN position using the tokenizer's offset mapping.

INPUTS
------
    data/telugu_math_train.json   (from 03_merge_and_clean.py)
    data/telugu_math_val.json     (from 03_merge_and_clean.py)
    Model downloaded from HuggingFace: google/muril-base-cased

OUTPUTS
-------
    models/muril_telugu_math/     ← saved fine-tuned model
        config.json
        pytorch_model.bin
        tokenizer files
    models/training_log.json      ← loss history for plotting

HOW TO RUN
----------
    Google Colab (REQUIRED — needs GPU):
        Runtime → Change runtime type → T4 GPU
        Then: !python 02_finetune.py

    ⚠️  Do NOT run on CPU — training will take hours.
        On Colab T4 GPU, expect 10–30 minutes for 3 epochs.

EXPECTED OUTPUT
---------------
    Epoch 1/3: train_loss=2.45, val_loss=2.31
    Epoch 2/3: train_loss=1.87, val_loss=1.94
    Epoch 3/3: train_loss=1.43, val_loss=1.71
    ✅ Model saved to models/muril_telugu_math/

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

# transformers provides the model, tokenizer, and Trainer
from transformers import (
    AutoTokenizer,                  # loads MuRIL tokenizer
    AutoModelForQuestionAnswering,  # loads MuRIL with QA head
    TrainingArguments,              # configures the training loop
    Trainer,                        # runs the training loop
    default_data_collator,          # batches examples together
)

# torch is PyTorch — the deep learning framework
import torch

# Dataset is HuggingFace's dataset class — needed by Trainer
from torch.utils.data import Dataset

# Standard Python libraries
import json
import os
import time

# =============================================================================
# CONSTANTS — all configuration in one place
# =============================================================================

# Pretrained model to fine-tune
MODEL_NAME = "google/muril-base-cased"

# Input data files from Week 7
# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------------------------
# __file__ does not work inside Jupyter/VSCode notebook cells.
# We detect the environment and set paths accordingly.
#
# If running as a .py script from terminal: use __file__ to auto-detect
# If running inside a notebook cell: use the known repo root directly
try:
    # Works when running as: python 03_model/02_finetune.py
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
except NameError:
    # __file__ is not defined in Jupyter/VSCode notebook cells
    # Set _REPO_ROOT to the telugu_math_chatbot folder directly
    _REPO_ROOT = os.path.expanduser(
        "~/Documents/lumiere_education/2026_siddharth/telugu_math_chatbot"
    )
    _SCRIPT_DIR = os.path.join(_REPO_ROOT, "03_model")

# Input data files — saved by 03_merge_and_clean.py into 01_data/
TRAIN_PATH = os.path.join(_REPO_ROOT, "data", "telugu_math_train.json")
VAL_PATH   = os.path.join(_REPO_ROOT, "data", "telugu_math_val.json")

# Where to save the fine-tuned model weights and tokenizer
MODEL_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "muril_telugu_math")

# Where to save training logs (loss curves for the paper)
LOG_DIR  = _SCRIPT_DIR
LOG_FILE = os.path.join(LOG_DIR, "training_log.json")

# Quick sanity check — print resolved paths so you can verify
print(f"📁 Repo root  : {_REPO_ROOT}")
print(f"📁 Train data : {TRAIN_PATH}")
print(f"📁 Val data   : {VAL_PATH}")
print(f"📁 Model out  : {MODEL_OUTPUT_DIR}")

# Training hyperparameters
# These are the standard starting values for BERT-style fine-tuning.
# We document these in the paper and compare two configs in Week 9.
LEARNING_RATE    = 2e-5   # standard for BERT fine-tuning (0.00002)
NUM_EPOCHS       = 100      # 3 epochs is standard for small datasets
BATCH_SIZE       = 8      # safe for Colab T4 GPU; reduce to 4 if OOM error
WEIGHT_DECAY     = 0.01   # L2 regularisation — helps prevent overfitting
WARMUP_STEPS     = 2000     # gradually increase LR at start of training

# Maximum token length for model input
# MuRIL supports 512 but we use 384 to fit question + context safely
MAX_LENGTH   = 384
DOC_STRIDE   = 128   # overlap when context is split across windows

# =============================================================================
# DATASET CLASS
# =============================================================================

class TeluguQADataset(Dataset):
    """
    A PyTorch Dataset for Telugu question answering.

    PyTorch's Trainer requires data in a specific format — a Dataset
    object that returns tokenized examples as dictionaries.

    This class:
        1. Loads our JSON data (list of dicts)
        2. Tokenizes each example using MuRIL tokenizer
        3. Converts character-level answer_start to token-level positions
        4. Returns properly formatted tensors for each example

    WHY CHARACTER → TOKEN CONVERSION?
        Our data stores answer positions as CHARACTER indices in the context
        (e.g. answer_start=85 means the answer starts at character 85).
        But the model works with TOKEN indices — each token may cover
        multiple characters. We must find which TOKEN corresponds to
        character 85 using the tokenizer's offset_mapping.

    HOW OFFSET MAPPING WORKS:
        The tokenizer returns offset_mapping — a list of (start, end) pairs,
        one per token, showing which characters each token covers.

        Example (simplified):
            context   : "Tom has 7 apples"
            tokens    : ['Tom', 'has', '7', 'ap', '##ples']
            offsets   : [(0,3), (4,7), (8,9), (10,12), (12,17)]

            answer_start = 8  (character position of "7")
            → We scan offsets to find token where start <= 8 < end
            → Token index 2 covers characters 8–9 → start_position = 2

    Args:
        examples  (list): Raw examples from JSON file.
        tokenizer       : Loaded MuRIL tokenizer.
        max_length (int): Maximum sequence length in tokens.
        doc_stride (int): Overlap when splitting long contexts.
    """

    def __init__(self, examples: list, tokenizer, max_length: int, doc_stride: int):
        self.examples   = examples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.features   = self._tokenize_all()

    def _tokenize_all(self) -> list:
        """
        Tokenize all examples and compute token-level answer positions.

        Returns:
            list: List of feature dicts ready for the Trainer.
        """
        features = []
        skipped  = 0

        for example in self.examples:
            feature = self._tokenize_one(example)
            if feature is not None:
                features.append(feature)
            else:
                skipped += 1

        print(f"   Tokenized : {len(features):,} examples")
        if skipped:
            print(f"   Skipped   : {skipped:,} (answer not found in token window)")

        return features

    def _tokenize_one(self, example: dict) -> dict:
        """
        Tokenize a single QA example and find token-level answer positions.

        The tokenizer takes BOTH question and context together:
            [CLS] question tokens [SEP] context tokens [SEP]

        We need token_type_ids=True so the model knows which tokens
        are the question (type 0) and which are the context (type 1).
        The answer must be in the context tokens only.

        Args:
            example (dict): One QA example with question, context,
                            answer_text, answer_start.

        Returns:
            dict: Tokenized feature with start_positions and end_positions,
                  or None if the answer cannot be located in the token window.
        """
        question = example["question"]
        context  = example["context"]
        answer   = example["answer_text"]
        char_start = example["answer_start"]
        char_end   = char_start + len(answer)

        # Tokenize question + context together
        # return_offsets_mapping=True gives us character↔token mapping
        # truncation="only_second" truncates the context (second sequence)
        # but never truncates the question (first sequence)
        encoding = self.tokenizer(
            question,
            context,
            max_length         = self.max_length,
            truncation         = "only_second",
            stride             = self.doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping    = True,
            padding            = "max_length",
            return_tensors     = "pt",
        )

        # Use only the first window (index 0)
        # For very long contexts, multiple windows are generated.
        # For simplicity we use only the first — answer must be within
        # the first MAX_LENGTH tokens. Documented as a limitation.
        input_ids      = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # sequence_ids() returns 0 for question tokens, 1 for context tokens,
        # None for special tokens ([CLS], [SEP]).
        # We only look for the answer in context tokens (sequence_id == 1).
        sequence_ids = encoding.sequence_ids(0)

        # Find the start and end token positions for the answer
        start_position = 0
        end_position   = 0
        found          = False

        for token_idx, (offset, seq_id) in enumerate(zip(offset_mapping, sequence_ids)):
            if seq_id != 1:
                # This token is from the question or is a special token — skip
                continue

            token_char_start, token_char_end = offset

            # Check if the answer span overlaps with this token
            if not found and token_char_start <= char_start < token_char_end:
                start_position = token_idx
                found = True

            if found and token_char_end >= char_end:
                end_position = token_idx
                break

        if not found:
            # Answer is outside the tokenized window — skip this example
            return None

        return {
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "start_positions": torch.tensor(start_position, dtype=torch.long),
            "end_positions"  : torch.tensor(end_position,   dtype=torch.long),
        }

    def __len__(self):
        """Return the number of tokenized features."""
        return len(self.features)

    def __getitem__(self, idx):
        """Return one tokenized feature by index."""
        return self.features[idx]


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data(path: str, label: str) -> list:
    """
    Load a JSON dataset file and return as a Python list.

    Args:
        path  (str): File path to the JSON file.
        label (str): Human-readable name for progress messages.

    Returns:
        list: The loaded examples, or empty list if file missing.
    """
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        print(f"   Run 03_merge_and_clean.py first (Week 7).")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {label}: {len(data):,} examples")
    return data


def check_device() -> str:
    """
    Check available hardware and return the best device string.

    Priority order:
        1. CUDA  — NVIDIA GPU (Colab T4, any NVIDIA card)
        2. MPS   — Apple Silicon GPU (Mac M1/M2/M3)
        3. CPU   — fallback (slow but works)

    For this project, MPS on a Mac M-series chip is fast enough
    for our small dataset (~370 training examples, 3 epochs).
    Expect ~5–15 minutes on MPS vs ~10–30 minutes on Colab T4.
    CPU will take 1–3 hours — not recommended but possible overnight.

    Returns:
        str: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🚀 NVIDIA GPU detected : {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   Training will be fast (~10–30 min).")
        return "cuda"

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"🍎 Apple MPS GPU detected (Mac M-series chip)")
        print(f"   Training will be moderately fast (~5–20 min).")
        return "mps"

    else:
        print(f"🖥️  No GPU detected — using CPU.")
        print(f"   Training will be slow (~1–3 hours for 3 epochs).")
        print(f"   Options to speed up:")
        print(f"     1. Open this notebook in Google Colab with T4 GPU")
        print(f"        (Runtime → Change runtime type → T4 GPU)")
        print(f"     2. If on a Mac M1/M2/M3, ensure PyTorch >= 2.0 is installed")
        print(f"   Continuing on CPU — this WILL work, just slowly.")
        return "cpu"


def build_training_args(output_dir: str, device: str) -> TrainingArguments:
    """
    Build the TrainingArguments object that configures the training loop.

    TrainingArguments is a HuggingFace class that holds all hyperparameters
    and settings for the Trainer. Think of it as the "settings panel"
    for fine-tuning.

    Key settings explained:
        num_train_epochs     : Train for 3 full passes through the data
        per_device_train_batch_size : 8 examples per GPU step
        learning_rate        : Step size for weight updates (2e-5)
        warmup_steps         : Gradually increase LR for first 50 steps
        weight_decay         : L2 regularisation (prevents overfitting)
        eval_strategy        : Evaluate on validation set after each epoch
        save_strategy        : Save model checkpoint after each epoch
        load_best_model_at_end : At the end, load the checkpoint with
                                 lowest validation loss (not last epoch)
        fp16                 : Use 16-bit floats on GPU (faster, less memory)

    Args:
        output_dir (str): Where to save model checkpoints.
        device     (str): "cuda" or "cpu".

    Returns:
        TrainingArguments: Configured training arguments object.
    """
    # fp16 (half precision) speeds up CUDA GPU training significantly.
    # Do NOT use on CPU (unsupported) or MPS (can cause instability).
    use_fp16 = (device == "cuda")

    # Batch size: use 8 on GPU (fast memory), 4 on MPS/CPU (less memory)
    effective_batch = BATCH_SIZE if device == "cuda" else 4

    args = TrainingArguments(
        output_dir                  = output_dir,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = effective_batch,
        per_device_eval_batch_size  = effective_batch,
        learning_rate               = LEARNING_RATE,
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy               = "epoch",   # evaluate after each epoch
        save_strategy               = "epoch",   # save checkpoint after each epoch
        load_best_model_at_end      = True,       # keep best checkpoint
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,      # lower loss = better
        logging_dir                 = os.path.join(output_dir, "logs"),
        logging_steps               = 10,         # print loss every 10 steps
        fp16                        = use_fp16,
        report_to                   = "none",     # disable wandb/mlflow logging
        save_total_limit            = 2,          # keep only last 2 checkpoints
    )

    print(f"\n  Training configuration:")
    print(f"    Epochs          : {NUM_EPOCHS}")
    print(f"    Batch size      : {effective_batch} ({'GPU mode' if device == 'cuda' else 'CPU/MPS mode'})")
    print(f"    Learning rate   : {LEARNING_RATE}")
    print(f"    Weight decay    : {WEIGHT_DECAY}")
    print(f"    Warmup steps    : {WARMUP_STEPS}")
    print(f"    Max token length: {MAX_LENGTH}")
    print(f"    Mixed precision : {'fp16 ✅' if use_fp16 else 'disabled (CPU/MPS)'}")

    return args


def save_training_log(trainer: Trainer) -> None:
    """
    Save the training loss history to a JSON file for plotting.

    The Trainer records loss at every logging step. We save this
    to disk so we can plot training vs validation loss curves —
    a standard figure in every NLP paper (shows the model is
    learning and not overfitting).

    Args:
        trainer (Trainer): The trained Trainer object.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    log_history = trainer.state.log_history

    with open(LOG_FILE, "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\n✅ Training log saved to: {LOG_FILE}")
    print(f"   Use this in Week 9 to plot loss curves for the paper.")

    # Print a summary of train and eval loss per epoch
    print(f"\n  Loss summary:")
    print(f"  {'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}")
    print(f"  {'------':>6}  {'----------':>12}  {'--------':>12}")

    train_losses = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_losses  = [e for e in log_history if "eval_loss" in e]

    for i, eval_entry in enumerate(eval_losses):
        epoch      = eval_entry.get("epoch", i + 1)
        eval_loss  = eval_entry.get("eval_loss", "N/A")
        # Get the training loss closest to this epoch
        train_loss = "N/A"
        for entry in reversed(train_losses):
            if entry.get("epoch", 0) <= epoch:
                train_loss = entry.get("loss", "N/A")
                break
        if isinstance(train_loss, float):
            train_loss = f"{train_loss:.4f}"
        if isinstance(eval_loss, float):
            eval_loss = f"{eval_loss:.4f}"
        print(f"  {epoch:>6.1f}  {train_loss:>12}  {eval_loss:>12}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point for the MuRIL fine-tuning script.

    Full pipeline:
        1. Check GPU availability
        2. Load training and validation data
        3. Load MuRIL tokenizer and model
        4. Tokenize datasets (convert text → token IDs + answer positions)
        5. Configure training hyperparameters
        6. Run fine-tuning with HuggingFace Trainer
        7. Save the fine-tuned model to disk
        8. Save training loss history for plotting

    This is Week 8 — the core of the project.
    After this script you have a trained Telugu math QA model.

    NOTE ON DATASET SIZE:
        Our dataset has ~462 examples after merging (300 MAWPS + 162 IndicQA).
        This is small by NLP standards. We expect moderate F1 scores (~40-60%)
        rather than state-of-the-art results. The contribution of this project
        is the methodology and the dataset construction — not beating benchmarks.
        We document this honestly in the paper's Discussion section.
    """
    print("=" * 60)
    print("Telugu Math Chatbot — Lumiere Research Project")
    print("Script: 02_finetune.py  |  Week 8")
    print("=" * 60)

    start_time = time.time()

    # -------------------------------------------------------------------------
    # STEP 1: Check hardware
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: HARDWARE CHECK")
    print("=" * 60)
    device = check_device()

    # -------------------------------------------------------------------------
    # STEP 2: Load data
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: LOADING DATA")
    print("=" * 60)
    train_data = load_data(TRAIN_PATH, "train")
    val_data   = load_data(VAL_PATH,   "validation")

    if not train_data:
        print("\n❌ No training data found. Exiting.")
        return

    # -------------------------------------------------------------------------
    # STEP 3: Load tokenizer and model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: LOADING MuRIL MODEL AND TOKENIZER")
    print("=" * 60)
    print(f"⏳ Loading tokenizer from: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✅ Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size:,}")

    print(f"\n⏳ Loading model from: {MODEL_NAME}")
    print(f"   (You will see a LOAD REPORT below — UNEXPECTED/MISSING keys are normal)")
    print(f"   (See Week 4 notes for explanation of why this is expected)\n")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model loaded. Parameters: {total_params:,}")

    # Move model to GPU if available
    model = model.to(device)
    print(f"   Model moved to: {device}")

    # -------------------------------------------------------------------------
    # STEP 4: Tokenize datasets
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: TOKENIZING DATASETS")
    print("=" * 60)
    print("Converting text → token IDs and finding answer token positions...")

    print(f"\n⏳ Tokenizing training set ({len(train_data):,} examples)...")
    train_dataset = TeluguQADataset(train_data, tokenizer, MAX_LENGTH, DOC_STRIDE)

    if val_data:
        print(f"\n⏳ Tokenizing validation set ({len(val_data):,} examples)...")
        val_dataset = TeluguQADataset(val_data, tokenizer, MAX_LENGTH, DOC_STRIDE)
    else:
        val_dataset = None
        print("⚠️  No validation data — training without evaluation.")

    print(f"\n  Final dataset sizes:")
    print(f"    Training   : {len(train_dataset):,} tokenized examples")
    if val_dataset:
        print(f"    Validation : {len(val_dataset):,} tokenized examples")

    # -------------------------------------------------------------------------
    # STEP 5: Configure training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: TRAINING CONFIGURATION")
    print("=" * 60)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    training_args = build_training_args(MODEL_OUTPUT_DIR, device)

    # -------------------------------------------------------------------------
    # STEP 6: Run fine-tuning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: FINE-TUNING")
    print("=" * 60)
    print(f"⏳ Starting training...")
    print(f"   This will take 10–30 minutes on GPU.")
    print(f"   Watch the loss numbers — they should decrease each epoch.\n")

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = default_data_collator,
    )

    # train() runs the full fine-tuning loop
    trainer.train()

    print(f"\n✅ Training complete!")

    # -------------------------------------------------------------------------
    # STEP 7: Save the fine-tuned model
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: SAVING MODEL")
    print("=" * 60)

    # Save model weights and config
    trainer.save_model(MODEL_OUTPUT_DIR)

    # Save tokenizer alongside the model — needed for inference
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print(f"✅ Model saved to  : {MODEL_OUTPUT_DIR}")
    print(f"   Files saved:")
    for f in os.listdir(MODEL_OUTPUT_DIR):
        size = os.path.getsize(os.path.join(MODEL_OUTPUT_DIR, f)) / 1e6
        print(f"     {f:40s} {size:.1f} MB")

    # -------------------------------------------------------------------------
    # STEP 8: Save training log
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: TRAINING LOG")
    print("=" * 60)
    save_training_log(trainer)

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — save these numbers for your paper!")
    print("=" * 60)
    print(f"  Base model      : {MODEL_NAME}")
    print(f"  Training examples: {len(train_dataset):,}")
    if val_dataset:
        print(f"  Validation examples: {len(val_dataset):,}")
    print(f"  Epochs          : {NUM_EPOCHS}")
    print(f"  Learning rate   : {LEARNING_RATE}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Max token length: {MAX_LENGTH}")
    print(f"  Total time      : {elapsed:.1f} minutes")
    print(f"  Model saved to  : {MODEL_OUTPUT_DIR}")
    print()
    print("  Next steps:")
    print("  → Week 9 : 01_evaluate_metrics.py")
    print("             Compute F1 Score and Exact Match on the test set")
    print("             Compare against the Week 4 baseline scores")
    print("  → Week 10: 01_gradio_app.py")
    print("             Build the chatbot demo interface")


# =============================================================================
# ENTRY POINT GUARD
# =============================================================================
if __name__ == "__main__":
    main()
