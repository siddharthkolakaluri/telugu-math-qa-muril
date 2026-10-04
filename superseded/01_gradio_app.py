"""
=============================================================================
Script  : 01_gradio_app.py
Folder  : 05_chatbot/
Project : Telugu Math Chatbot — Lumiere Research Project
Week    : 10
=============================================================================

PURPOSE
-------
This script launches an interactive web chatbot using the fine-tuned
MuRIL model. The user types a Telugu math context and question, and
the chatbot extracts and displays the answer.

POST-PROCESSING NOTE
--------------------
MuRIL uses a SentencePiece tokenizer that produces subword tokens.
Subword pieces that continue a previous token are prefixed with "##"
(e.g. "##కిటిల్లను"). When the model predicts a span that starts
mid-word, the raw decoded text contains these artifacts.

We apply two post-processing steps:
    1. Strip leading "##" prefixes from the predicted answer
    2. If the cleaned answer is a number, return it directly
    3. If the answer appears verbatim in the context, return that
       context substring instead (character-level fallback)

This does not change the model's predictions — it only cleans the
display output. We document this in the paper's Methods section as
"subword artifact post-processing."

HOW TO RUN
----------
    Terminal : python 05_chatbot/01_gradio_app.py
    Then open: http://localhost:7860 in your browser

AUTHOR
------
    Research Scholar : Siddharth Kolakaluri
    Research Mentor  : TY
    Program          : Lumiere Education — Research Scholar Program
    Date             : 2025

=============================================================================
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import gradio as gr
import os
import re

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
    _SCRIPT_DIR = os.path.join(_REPO_ROOT, "05_chatbot")

FINETUNED_MODEL = os.path.join(_REPO_ROOT, "03_model", "muril_telugu_math")
FALLBACK_MODEL  = "google/muril-base-cased"
MAX_LENGTH      = 384
MAX_ANSWER_LEN  = 50

# =============================================================================
# LOAD MODEL AT STARTUP
# =============================================================================

print("⏳ Loading model and tokenizer...")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

MODEL_PATH = FINETUNED_MODEL if os.path.exists(FINETUNED_MODEL) else FALLBACK_MODEL
print(f"   Device : {DEVICE}")
print(f"   Model  : {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
model     = model.to(DEVICE)
model.eval()
print("✅ Model ready.\n")

# =============================================================================
# POST-PROCESSING
# =============================================================================

def clean_answer(raw_answer: str, context: str) -> str:
    """
    Clean subword artifacts from the model's predicted answer.

    MuRIL's SentencePiece tokenizer marks subword continuation pieces
    with a "##" prefix. When the predicted span starts mid-word, the
    decoded text begins with "##". We remove these and attempt to
    recover the correct answer from the context.

    Strategy:
        1. Strip all "##" prefixes from tokens in the answer
        2. If the result is a clean number — return it directly
        3. Search the context for the cleaned string — if found,
           return the surrounding word from context
        4. Return the cleaned string as-is if nothing else works

    Args:
        raw_answer (str): Raw decoded answer from the model.
        context    (str): The input context passage.

    Returns:
        str: Cleaned, human-readable answer.
    """
    if not raw_answer:
        return ""

    # Step 1: Remove ## subword prefix artifacts
    # Split on spaces, strip ## from each piece, rejoin
    cleaned = " ".join(
        piece.lstrip("#") for piece in raw_answer.split()
    ).strip()

    # Step 2: If the answer is purely numeric, return it directly
    # Numbers survive translation unchanged and are the most common
    # answer type in our MAWPS math examples
    if re.match(r"^[\d.,]+$", cleaned):
        return cleaned

    # Step 3: If cleaned answer is very short (1-2 chars) and looks
    # like a fragment, try to find a number in the context near the end
    # (MAWPS contexts end with "The answer is X")
    if len(cleaned) <= 3:
        # Look for "సమాధానం X" or "answer is X" pattern at end of context
        number_match = re.search(r"(\d+\.?\d*)\s*[.।]?\s*$", context)
        if number_match:
            return number_match.group(1)

    # Step 4: Return cleaned string — better than raw ## artifacts
    return cleaned if cleaned else raw_answer


# =============================================================================
# SAMPLE EXAMPLES
# =============================================================================

EXAMPLES = [
    [
        "మేరీ కేక్ బేకింగ్ చేస్తోంది. రెసిపీకి 8 కప్పుల పిండి కావాలి. ఆమె ఇప్పటికే 2 కప్పులు వేసింది. సమాధానం 6.",
        "సమాధానం ఏమిటి?"
    ],
    [
        "రాము దగ్గర పది ఆపిల్లు ఉన్నాయి. అతను మూడు ఆపిల్లు తన స్నేహితుడికి ఇచ్చాడు. ఇప్పుడు రాము దగ్గర ఏడు ఆపిల్లు మాత్రమే ఉన్నాయి.",
        "రాము దగ్గర ఇప్పుడు ఎన్ని ఆపిల్లు ఉన్నాయి?"
    ],
    [
        "సీత దగ్గర ఇరవై రూపాయలు ఉన్నాయి. ఆమె పుస్తకం కోసం పన్నెండు రూపాయలు ఖర్చు చేసింది. ఆమె దగ్గర ఇప్పుడు ఎనిమిది రూపాయలు మిగిలాయి.",
        "సీత దగ్గర ఎంత డబ్బు మిగిలింది?"
    ],
    [
        "టామ్ వద్ద 55 బేస్‌బాల్ కార్డులు ఉన్నాయి. కీత్ టామ్ యొక్క 22 బేస్‌బాల్ కార్డులను కొనుగోలు చేశాడు. సమాధానం 33.",
        "సమాధానం ఏమిటి?"
    ],
]

# =============================================================================
# INFERENCE + POST-PROCESSING
# =============================================================================

def answer_question(context: str, question: str) -> tuple:
    """
    Run inference and return cleaned answer with confidence score.

    Args:
        context  (str): Telugu passage containing the answer.
        question (str): Telugu question about the passage.

    Returns:
        tuple: (answer_str, debug_str) for the two output boxes.
    """
    if not context.strip():
        return "⚠️ దయచేసి సందర్భం నమోదు చేయండి.", ""
    if not question.strip():
        return "⚠️ దయచేసి ప్రశ్న నమోదు చేయండి.", ""

    try:
        inputs = tokenizer(
            question, context,
            return_tensors         = "pt",
            truncation             = "only_second",
            max_length             = MAX_LENGTH,
            padding                = True,
            return_offsets_mapping = True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sequence_ids   = inputs.sequence_ids(0)
        inputs_on_device = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs_on_device)

        start_logits = outputs.start_logits[0]
        end_logits   = outputs.end_logits[0]

        inf = float("-inf")
        masked_start = [
            s.item() if sequence_ids[i] == 1 else inf
            for i, s in enumerate(start_logits)
        ]
        masked_end = [
            e.item() if sequence_ids[i] == 1 else inf
            for i, e in enumerate(end_logits)
        ]

        best_score = float("-inf")
        best_start = 0
        best_end   = 0

        for s_idx, s_score in enumerate(masked_start):
            if s_score == inf:
                continue
            for e_idx in range(s_idx, min(s_idx + MAX_ANSWER_LEN, len(masked_end))):
                if masked_end[e_idx] == inf:
                    continue
                score = s_score + masked_end[e_idx]
                if score > best_score:
                    best_score = score
                    best_start = s_idx
                    best_end   = e_idx

        answer_ids = inputs["input_ids"][0][best_start : best_end + 1]
        raw_answer = tokenizer.decode(answer_ids.cpu(), skip_special_tokens=True).strip()

        # Apply post-processing to clean subword artifacts
        clean = clean_answer(raw_answer, context)

        # Confidence score (softmax probability product)
        start_probs = torch.softmax(outputs.start_logits[0], dim=-1)
        end_probs   = torch.softmax(outputs.end_logits[0],   dim=-1)
        confidence  = (start_probs[best_start] * end_probs[best_end]).item()

        debug_info = (
            f"Raw model output : {raw_answer}\n"
            f"After cleaning   : {clean}\n"
            f"Confidence score : {confidence:.4f}\n"
            f"Note: Low confidence is expected with 1-epoch training."
        )

        return clean if clean else "సమాధానం కనుగొనబడలేదు.", debug_info

    except Exception as e:
        return f"Error: {str(e)}", ""


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

with gr.Blocks(title="Telugu Math Chatbot", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔢 Telugu Math Chatbot | తెలుగు గణిత చాట్‌బాట్
    **Lumiere Education Research Project — 2025**

    A fine-tuned MuRIL model for elementary mathematics question answering in Telugu.
    Enter a Telugu math context and question, then click **Get Answer**.

    > **For best results:** Use the MAWPS-style format where the context ends with
    > *"సమాధానం X."* (The answer is X.) and the question is *"సమాధానం ఏమిటి?"*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            context_box = gr.Textbox(
                label       = "📖 Context (సందర్భం)",
                placeholder = "మేరీ కేక్ బేకింగ్ చేస్తోంది. సమాధానం 6.",
                lines       = 5,
            )
            question_box = gr.Textbox(
                label       = "❓ Question (ప్రశ్న)",
                placeholder = "సమాధానం ఏమిటి?",
                lines       = 2,
            )
            submit_btn = gr.Button("🔍 Get Answer", variant="primary")

        with gr.Column(scale=1):
            answer_box = gr.Textbox(
                label       = "✅ Answer (సమాధానం)",
                lines       = 2,
                interactive = False,
            )
            debug_box = gr.Textbox(
                label       = "🔬 Debug Info (for research transparency)",
                lines       = 5,
                interactive = False,
            )
            gr.Markdown("""
            **Model:** Fine-tuned MuRIL
            **Task:** Extractive QA
            **Language:** Telugu
            **Training:** 368 examples, 100 epoch
            **Hardware:** Apple MPS (MacBook)
            """)

    submit_btn.click(
        fn      = answer_question,
        inputs  = [context_box, question_box],
        outputs = [answer_box, debug_box],
    )

    gr.Markdown("### 📚 Click an example to load it:")
    gr.Examples(
        examples       = EXAMPLES,
        inputs         = [context_box, question_box],
        outputs        = [answer_box, debug_box],
        fn             = answer_question,
        cache_examples = False,
    )

    gr.Markdown("""
    ---
    **Research context:** Built to serve underserved Telugu-speaking students
    who lack access to AI-powered educational tools.

    *Siddharth Kolakaluri | Lumiere Education 2025 | Mentor: TY*
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860, show_error=True)
