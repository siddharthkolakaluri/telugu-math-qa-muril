"""
=============================================================================
Script  : 03_gradio_app_v3.py
Folder  : chatbot/
Project : Telugu Math QA — fine-tuned MuRIL, extractive span prediction
=============================================================================

WHAT THIS IS
------------
A workbench for the fine-tuned MuRIL model. Left rail holds the session's
query history, the examples, and the model card. The canvas holds a running
transcript: each query shows the passage with the model's selected span
marked in place, and the answer it produced.

A NOTE ON THE TRANSCRIPT
------------------------
This looks like a chat, but the model has no memory between queries. Every
entry is an independent passage-plus-question prediction. The transcript is
a lab log, not a conversation — a follow-up question will be answered from
whatever passage you give it, not from anything above.

HOW TO RUN
----------
    conda activate nlp
    python chatbot/03_gradio_app_v3.py
    then open http://localhost:7860

AUTHOR
------
    Siddharth Kolakaluri — CHIREC International School
    Lumiere Education Research Scholar Program
=============================================================================
"""

import os
import re
import html
import argparse

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# =============================================================================
# CONFIG
# =============================================================================

MAX_LENGTH = 384
MAX_ANSWER_LEN = 50
FALLBACK_MODEL = "google/muril-base-cased"

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
except NameError:
    _REPO_ROOT = os.getcwd()
    _SCRIPT_DIR = os.path.join(_REPO_ROOT, "chatbot")

# The run directory comes before its checkpoint-NNNN subfolders: training used
# load_best_model_at_end, so the run directory holds the best weights AND the
# tokenizer, while a checkpoint subfolder has weights and optimizer state but
# no tokenizer files.
MODEL_CANDIDATES = [
    os.path.join(_REPO_ROOT, "model", "muril_telugu_math_v2"),
    os.path.join(_REPO_ROOT, "model", "muril_telugu_math_v2", "checkpoint-5008"),
    os.path.join(_REPO_ROOT, "model", "muril_seed42"),
    os.path.join(_REPO_ROOT, "model", "muril_telugu_math"),
]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "sentencepiece.bpe.model")


def _has(path, names):
    return any(os.path.exists(os.path.join(path, n)) for n in names)


def resolve_model_path(explicit=None):
    for path in ([explicit] if explicit else []) + MODEL_CANDIDATES:
        if path and os.path.isdir(path) and \
                _has(path, ("model.safetensors", "pytorch_model.bin")):
            return path, True
    return FALLBACK_MODEL, False


def resolve_tokenizer_path(model_path, is_finetuned):
    if not is_finetuned or _has(model_path, TOKENIZER_FILES):
        return model_path
    parent = os.path.dirname(model_path.rstrip(os.sep))
    if parent and os.path.isdir(parent) and _has(parent, TOKENIZER_FILES):
        return parent
    return FALLBACK_MODEL


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


parser = argparse.ArgumentParser(description="Telugu math QA workbench")
parser.add_argument("--model", default=None)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
ARGS, _ = parser.parse_known_args()

MODEL_PATH, IS_FINETUNED = resolve_model_path(ARGS.model)
TOKENIZER_PATH = resolve_tokenizer_path(MODEL_PATH, IS_FINETUNED)
DEVICE = pick_device()

print("Loading model...")
print(f"  weights   : {MODEL_PATH}")
print(f"  tokenizer : {TOKENIZER_PATH}")
print(f"  tuned     : {IS_FINETUNED}")
print(f"  device    : {DEVICE}")
if not IS_FINETUNED:
    print("  WARNING: fine-tuned weights not found. Running the untrained")
    print("           base model. Predictions will be meaningless.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Ready.\n")


# =============================================================================
# POST-PROCESSING
# =============================================================================

def clean_answer(raw_answer, context):
    """
    Returns (text, method). Method is one of:
        span     the model's span with ## subword markers stripped
        numeric  the span was already a bare number
        pattern  the span was unusable, so the trailing number in the passage
                 was substituted. This is a heuristic, NOT the model.
        empty    nothing usable
    """
    if not raw_answer:
        return "", "empty"

    cleaned = " ".join(p.lstrip("#") for p in raw_answer.split()).strip()

    if re.match(r"^[\d.,]+$", cleaned):
        return cleaned, "numeric"

    if len(cleaned) <= 3:
        # A decimal point is only part of the number when digits follow it,
        # otherwise "సమాధానం 6." yields "6." instead of "6".
        trailing = re.search(r"(\d+(?:\.\d+)?)\s*[.।]?\s*$", context)
        if trailing:
            return trailing.group(1), "pattern"

    return (cleaned or raw_answer), "span"


# =============================================================================
# INFERENCE
# =============================================================================

def predict(context, question):
    enc = tokenizer(
        question, context,
        return_tensors="pt",
        truncation="only_second",
        max_length=MAX_LENGTH,
        padding=True,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0]
    sequence_ids = enc.sequence_ids(0)
    on_device = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**on_device)

    start_logits = out.start_logits[0].float().cpu()
    end_logits = out.end_logits[0].float().cpu()
    length = start_logits.size(0)

    context_mask = torch.tensor([sid == 1 for sid in sequence_ids],
                                dtype=torch.bool)
    if not context_mask.any():
        return {"error": "The passage did not survive tokenisation. "
                         "Try a shorter passage."}

    neg = torch.finfo(start_logits.dtype).min
    s = start_logits.masked_fill(~context_mask, neg)
    e = end_logits.masked_fill(~context_mask, neg)

    scores = s.unsqueeze(1) + e.unsqueeze(0)
    idx = torch.arange(length)
    valid = (idx.unsqueeze(0) >= idx.unsqueeze(1)) & \
            ((idx.unsqueeze(0) - idx.unsqueeze(1)) < MAX_ANSWER_LEN)
    scores = scores.masked_fill(~valid, neg)

    flat = int(torch.argmax(scores))
    best_start, best_end = flat // length, flat % length

    char_start = int(offsets[best_start][0])
    char_end = int(offsets[best_end][1])

    raw_decoded = tokenizer.decode(
        enc["input_ids"][0][best_start:best_end + 1],
        skip_special_tokens=True,
    ).strip()

    cleaned, method = clean_answer(raw_decoded, context)

    start_probs = torch.softmax(s, dim=-1)
    end_probs = torch.softmax(e, dim=-1)
    confidence = float(start_probs[best_start] * end_probs[best_end])

    return {
        "char_start": char_start,
        "char_end": char_end,
        "span_text": context[char_start:char_end],
        "raw_decoded": raw_decoded,
        "answer": cleaned,
        "method": method,
        "confidence": confidence,
        "token_start": best_start,
        "token_end": best_end,
        "truncated": int(context_mask.sum()) < len(tokenizer.tokenize(context)),
    }


# =============================================================================
# RENDERING
# =============================================================================

EMPTY_CANVAS = """
<div class="empty">
  <div class="empty-mark">◆</div>
  <p>Give the model a Telugu passage and a question about it.</p>
  <p class="empty-sub">It answers by marking a span of your passage. Pick an
     example from the rail to see what that looks like.</p>
</div>
"""

METHOD_NOTE = {
    "pattern": ("This answer did not come from the model. Its span was too "
                "short to use, so the trailing number in the passage was "
                "substituted — the same rule a non-learning extractor applies. "
                "The marked span is the model's real prediction."),
    "empty": "The model returned an empty span.",
}


def band_for(pct):
    if pct >= 50:
        return "high"
    if pct >= 15:
        return "moderate"
    return "low"


def marked_passage(context, cs, ce):
    if ce > cs:
        return (html.escape(context[:cs])
                + '<mark>' + html.escape(context[cs:ce]) + '</mark>'
                + html.escape(context[ce:]))
    return html.escape(context)


def render_turn(turn, index):
    if turn.get("error"):
        return (
            f'<article class="turn">'
            f'  <div class="q">{html.escape(turn["question"])}</div>'
            f'  <div class="err">{html.escape(turn["error"])}</div>'
            f'</article>'
        )

    pct = turn["confidence"] * 100
    note = METHOD_NOTE.get(turn["method"], "")
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    trunc = ('<div class="note">The passage exceeded the 384-token window and '
             'was cut off; the answer may lie in the removed part.</div>'
             if turn["truncated"] else "")

    return f"""
    <article class="turn">
      <div class="turn-n">{index}</div>
      <div class="q">{html.escape(turn["question"])}</div>
      <div class="passage">{marked_passage(turn["context"],
                                           turn["char_start"],
                                           turn["char_end"])}</div>
      <div class="a-row">
        <div class="a">{html.escape(turn["answer"]) or "—"}</div>
        <div class="score score-{band_for(pct)}">
          <span class="score-bar"><i style="width:{min(pct,100):.0f}%"></i></span>
          <span class="score-n">{pct:.1f}% {band_for(pct)}</span>
        </div>
      </div>
      {note_html}{trunc}
      <details class="internals">
        <summary>Model internals</summary>
        <dl>
          <dt>Span, decoded</dt><dd>{html.escape(turn["raw_decoded"]) or "—"}</dd>
          <dt>Span, from passage</dt><dd>{html.escape(turn["span_text"]) or "—"}</dd>
          <dt>Displayed answer</dt><dd>{html.escape(turn["answer"]) or "—"}</dd>
          <dt>Produced by</dt><dd>{turn["method"]}</dd>
          <dt>Character offsets</dt><dd>{turn["char_start"]}–{turn["char_end"]}</dd>
          <dt>Token indices</dt><dd>{turn["token_start"]}–{turn["token_end"]}</dd>
          <dt>Span score</dt><dd>{turn["confidence"]:.6f}</dd>
        </dl>
        <p>Span score is the product of the start and end softmax
           probabilities. It is not calibrated and is not the probability that
           the answer is correct.</p>
      </details>
    </article>
    """


def render_transcript(turns):
    if not turns:
        return EMPTY_CANVAS
    return ('<div class="transcript">'
            + "".join(render_turn(t, i + 1) for i, t in enumerate(turns))
            + '</div>')


def history_labels(turns):
    out = []
    for i, t in enumerate(turns):
        q = t["question"].strip() or "(no question)"
        if len(q) > 38:
            q = q[:37] + "…"
        out.append(f"{i + 1}. {q}")
    return out


# =============================================================================
# EXAMPLES
# =============================================================================
# The first two are MAWPS-style: the passage ends with the answer already
# stated, so locating it needs no arithmetic. This is the subset where a
# non-learning extractor also scores 100.0. The last two are comprehension
# style, where the answer sits inside the passage rather than at a fixed
# position — the case the reported 57.7 F1 refers to.

EXAMPLES = [
    ("Template · flour",
     "మేరీ కేక్ బేకింగ్ చేస్తోంది. రెసిపీకి 8 కప్పుల పిండి కావాలి. "
     "ఆమె ఇప్పటికే 2 కప్పులు వేసింది. సమాధానం 6.",
     "సమాధానం ఏమిటి?"),
    ("Template · cards",
     "టామ్ వద్ద 55 బేస్‌బాల్ కార్డులు ఉన్నాయి. కీత్ టామ్ యొక్క 22 "
     "బేస్‌బాల్ కార్డులను కొనుగోలు చేశాడు. సమాధానం 33.",
     "సమాధానం ఏమిటి?"),
    ("Comprehension · apples",
     "రాము దగ్గర పది ఆపిల్లు ఉన్నాయి. అతను మూడు ఆపిల్లు తన "
     "స్నేహితుడికి ఇచ్చాడు. ఇప్పుడు రాము దగ్గర ఏడు ఆపిల్లు మాత్రమే ఉన్నాయి.",
     "రాము దగ్గర ఇప్పుడు ఎన్ని ఆపిల్లు ఉన్నాయి?"),
    ("Comprehension · rupees",
     "సీత దగ్గర ఇరవై రూపాయలు ఉన్నాయి. ఆమె పుస్తకం కోసం పన్నెండు "
     "రూపాయలు ఖర్చు చేసింది. ఆమె దగ్గర ఇప్పుడు ఎనిమిది రూపాయలు మిగిలాయి.",
     "సీత దగ్గర ఎంత డబ్బు మిగిలింది?"),
]
EXAMPLE_LABELS = [e[0] for e in EXAMPLES]


# =============================================================================
# EVENTS
# =============================================================================

def on_submit(context, question, turns):
    context = (context or "").strip()
    question = (question or "").strip()
    turns = list(turns or [])

    if not context or not question:
        missing = "a passage" if not context else "a question"
        return (render_transcript(turns), turns,
                gr.update(choices=history_labels(turns), value=None),
                context, question,
                f"Add {missing} before searching.")

    try:
        r = predict(context, question)
    except Exception as exc:
        r = {"error": f"The model could not run: {exc}"}

    turn = {"context": context, "question": question}
    turn.update(r)
    turns.append(turn)

    return (render_transcript(turns), turns,
            gr.update(choices=history_labels(turns), value=None),
            context, "", "")


def on_pick_example(label, turns):
    for lab, ctx, q in EXAMPLES:
        if lab == label:
            return ctx, q, ""
    return gr.update(), gr.update(), ""


def on_pick_history(label, turns):
    turns = list(turns or [])
    if not label:
        return gr.update(), gr.update()
    try:
        i = int(label.split(".", 1)[0]) - 1
    except ValueError:
        return gr.update(), gr.update()
    if 0 <= i < len(turns):
        return turns[i]["context"], turns[i]["question"]
    return gr.update(), gr.update()


def on_new():
    return "", "", gr.update(value=None), ""


def on_clear_history():
    return (render_transcript([]), [],
            gr.update(choices=[], value=None), "")


# =============================================================================
# STYLES
# =============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Telugu:wght@400;600&display=swap');

/* Every surface is driven by these two sets. The theme is chosen by
   data-ui-theme on <html>, set from localStorage on load, so Gradio's own
   OS-following theme never half-applies underneath. */
:root, :root[data-ui-theme="light"] {
  color-scheme: light;
  --canvas:      #FFFFFF;
  --rail:        #F7F7F5;
  --raise:       #FFFFFF;
  --sunken:      #FCFCFB;
  --ink:         #1F2328;
  --ink-2:       #6B7280;
  --ink-3:       #9CA3AF;
  --line:        #E6E6E3;
  --line-soft:   #F0F0ED;
  --rail-hover:  #EEEEEB;
  --rail-active: #EAEAE6;
  --chip:        #EEEEEB;
  --mark:        #FFE8A3;
  --mark-text:   #1F2328;
  --mark-edge:   #D9A400;
  --good:        #2F7D4F;
  --warn-bg:     #FBEAE4;
  --warn-edge:   #B3472E;
  --warn-ink:    #7A2E1B;
  --scroll:      #E0E0DC;
  --scroll-hi:   #CFCFC9;
  --btn-hover:   #000000;
  --shadow:      rgba(0,0,0,.04);
}

:root[data-ui-theme="dark"] {
  color-scheme: dark;
  --canvas:      #1A1D22;
  --rail:        #14171B;
  --raise:       #22262C;
  --sunken:      #1E2127;
  --ink:         #E8EAED;
  --ink-2:       #A8AEB8;
  --ink-3:       #6E7681;
  --line:        #2A2E35;
  --line-soft:   #22262C;
  --rail-hover:  #1F232A;
  --rail-active: #262B32;
  --chip:        #252A31;
  --mark:        #4A3A10;
  --mark-text:   #FFDF8A;
  --mark-edge:   #D9A400;
  --good:        #5BAE7C;
  --warn-bg:     #3A211A;
  --warn-edge:   #C2563A;
  --warn-ink:    #F0B9A6;
  --scroll:      #313740;
  --scroll-hi:   #3D444E;
  --btn-hover:   #FFFFFF;
  --shadow:      rgba(0,0,0,.3);
}

html, body, .gradio-container, .dark {
  background: var(--canvas) !important;
  color: var(--ink) !important;
}

/* Gradio sets overflow:hidden on its container and drives heights through a
   nested flex chain. Fixed heights inside that chain cannot scroll, so the
   chain is opened up and the document does the scrolling instead. */
html, body { height: auto !important; overflow-y: auto !important; }
gradio-app { display: block !important; }
.gradio-container,
.gradio-container > .main,
.gradio-container .main > .wrap,
.gradio-container main.contain {
  height: auto !important; max-height: none !important;
  overflow: visible !important;
}
.gradio-container {
  max-width: 100% !important; width: 100% !important;
  margin: 0 !important; padding: 0 !important;
  font-size: 14px;
}
/* Gradio insets its content; this app is full-bleed. */
.gradio-container > .main,
.gradio-container .main > .wrap,
.gradio-container main.contain,
.gradio-container .wrap > .contain {
  padding: 0 !important; margin: 0 !important;
  max-width: none !important; width: 100% !important;
}
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
footer { display: none !important; }

/* ---- shell ------------------------------------------------------------- */
#shell { gap: 0 !important; align-items: flex-start !important; }

#rail {
  background: var(--rail) !important;
  border-right: 1px solid var(--line);
  padding: 20px 14px !important;
  min-width: 248px; max-width: 272px;
  gap: 4px !important;
  position: sticky; top: 0;
  min-height: 100vh; max-height: 100vh; overflow-y: auto;
}
#canvas {
  padding: 0 !important; min-width: 0;
  display: flex !important; flex-direction: column !important;
  flex-wrap: nowrap !important; min-height: 100vh;
}

/* ---- rail -------------------------------------------------------------- */
#brand { margin-bottom: 16px; }
#brand .bname {
  font-size: .9375rem; font-weight: 600; color: var(--ink);
  letter-spacing: -.01em; line-height: 1.3;
}
#brand .bsub { font-size: .75rem; color: var(--ink-3); margin-top: 2px; }

.rail-h {
  font-size: .6875rem; font-weight: 600; color: var(--ink-3);
  margin: 18px 0 6px 6px;
}

#newq, #newq button {
  width: 100%; justify-content: flex-start !important;
  background: var(--raise) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 7px !important;
  font-weight: 500 !important; font-size: .8125rem !important;
  padding: 8px 11px !important; box-shadow: 0 1px 1px var(--shadow) !important;
}
#newq:hover, #newq button:hover { background: var(--sunken) !important; }

/* rail lists: radios restyled as navigation rows */
#history .wrap, #examples .wrap { gap: 1px !important; }
#history label, #examples label {
  display: block !important; width: 100%;
  background: transparent !important; border: none !important;
  padding: 7px 10px !important; border-radius: 6px !important;
  cursor: pointer; transition: background .12s;
}
#history label:hover, #examples label:hover { background: var(--rail-hover) !important; }
#history input[type="radio"], #examples input[type="radio"] { display: none !important; }
#history label span, #examples label span {
  font-size: .8125rem !important; color: var(--ink-2) !important;
  font-weight: 450 !important; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; display: block;
}
#history label:has(input:checked), #examples label:has(input:checked) {
  background: var(--rail-active) !important;
}
#history label:has(input:checked) span,
#examples label:has(input:checked) span { color: var(--ink) !important; }

#clearh, #clearh button {
  background: transparent !important; border: none !important;
  color: var(--ink-3) !important; font-size: .75rem !important;
  padding: 4px 10px !important; justify-content: flex-start !important;
}
#clearh:hover, #clearh button:hover { color: var(--warn-edge) !important; }

#themebtn { margin-top: auto !important; }
#themebtn, #themebtn button {
  width: 100%; justify-content: flex-start !important;
  background: transparent !important; border: 1px solid var(--line) !important;
  color: var(--ink-2) !important; border-radius: 7px !important;
  font-size: .75rem !important; font-weight: 500 !important;
  padding: 7px 11px !important; box-shadow: none !important;
}
#themebtn:hover, #themebtn button:hover {
  background: var(--rail-hover) !important; color: var(--ink) !important;
}
#themebtn:focus-visible, #themebtn button:focus-visible {
  outline: 2px solid var(--mark-edge); outline-offset: 2px;
}

#card {
  padding-top: 14px; margin-top: 14px; border-top: 1px solid var(--line);
  font-size: .6875rem; color: var(--ink-3); line-height: 1.75;
}
#card b { color: var(--ink-2); font-weight: 600; }
#card code {
  font-size: .625rem; background: var(--chip); padding: 1px 4px;
  border-radius: 3px; color: var(--ink-2);
}

/* ---- canvas ------------------------------------------------------------ */
#scroll {
  flex: 1 1 auto;
  padding: 26px 32px 24px 32px;
}
#scroll > div { max-width: 740px; margin: 0 auto; }

.empty { text-align: center; padding: 16vh 0 0 0; color: var(--ink-3); }
.empty-mark { font-size: 1.1rem; color: var(--mark-edge); margin-bottom: 14px; }
.empty p { margin: 0 0 6px 0; font-size: .9375rem; color: var(--ink-2); }
.empty .empty-sub {
  font-size: .8125rem; color: var(--ink-3);
  max-width: 44ch; margin: 0 auto; line-height: 1.6;
}

.turn { position: relative; padding: 0 0 30px 0; margin-bottom: 30px;
        border-bottom: 1px solid var(--line-soft); }
.turn:last-child { border-bottom: none; }
.turn-n {
  position: absolute; left: -26px; top: 2px;
  font-size: .6875rem; color: var(--ink-3);
}

.q {
  font-family: 'Noto Serif Telugu', serif; font-size: 1rem; font-weight: 600;
  color: var(--ink); line-height: 1.85; margin-bottom: 14px;
}

.passage {
  font-family: 'Noto Serif Telugu', serif; font-size: .9375rem;
  line-height: 2.05; color: var(--ink-2);
  background: var(--sunken); border: 1px solid var(--line);
  border-left: 2px solid var(--ink); border-radius: 0 6px 6px 0;
  padding: 13px 16px; margin-bottom: 14px;
}
.passage mark {
  background: var(--mark); color: var(--mark-text);
  box-shadow: inset 0 -2px 0 var(--mark-edge);
  border-radius: 2px; padding: .06em .16em;
}

.a-row { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.a {
  font-family: 'Noto Serif Telugu', serif; font-size: 1.5rem;
  font-weight: 600; color: var(--ink); line-height: 1.5;
}
.score { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.score-bar {
  display: inline-block; width: 54px; height: 3px;
  background: var(--line); border-radius: 2px; overflow: hidden;
}
.score-bar i { display: block; height: 100%; background: var(--ink-3); }
.score-high .score-bar i { background: var(--good); }
.score-moderate .score-bar i { background: var(--mark-edge); }
.score-low .score-bar i { background: var(--ink-3); }
.score-n { font-size: .6875rem; color: var(--ink-3); }

.note {
  margin-top: 12px; padding: 9px 12px; border-radius: 6px;
  background: var(--warn-bg); border-left: 2px solid var(--warn-edge);
  color: var(--warn-ink); font-size: .75rem; line-height: 1.65;
}
.err {
  padding: 9px 12px; border-radius: 6px; background: var(--warn-bg);
  color: var(--warn-ink); font-size: .8125rem;
}

.internals { margin-top: 14px; }
.internals summary {
  cursor: pointer; font-size: .75rem; color: var(--ink-3);
  list-style: none; user-select: none; display: inline-block;
}
.internals summary::-webkit-details-marker { display: none; }
.internals summary::before { content: "▸ "; }
.internals[open] summary::before { content: "▾ "; }
.internals summary:hover { color: var(--ink-2); }
.internals dl {
  margin: 10px 0 0 0; display: grid;
  grid-template-columns: 150px 1fr; gap: 3px 14px;
  font-size: .75rem; line-height: 1.7;
}
.internals dt { color: var(--ink-3); }
.internals dd {
  margin: 0; color: var(--ink-2);
  font-family: ui-monospace, Menlo, monospace; font-size: .6875rem;
  word-break: break-word;
}
.internals p {
  margin: 10px 0 0 0; font-size: .6875rem;
  color: var(--ink-3); line-height: 1.65; max-width: 62ch;
}

/* ---- composer ---------------------------------------------------------- */
#composer {
  flex: 0 0 auto; position: sticky; bottom: 0; z-index: 5;
  border-top: 1px solid var(--line); background: var(--canvas);
  padding: 14px 32px 18px 32px;
}
#composer > div { max-width: 740px; margin: 0 auto; }

#cbox {
  border: 1px solid var(--line) !important; border-radius: 12px !important;
  background: var(--sunken) !important; padding: 4px !important;
  box-shadow: 0 1px 3px var(--shadow) !important;
}
#cbox:focus-within {
  border-color: var(--ink-3) !important;
  box-shadow: 0 1px 4px var(--shadow) !important;
}

#pbox textarea, #qbox textarea {
  font-family: 'Noto Serif Telugu', serif !important;
  background: transparent !important; border: none !important;
  box-shadow: none !important; resize: none !important;
  color: var(--ink) !important; padding: 7px 10px !important;
}
#pbox textarea { font-size: .9375rem !important; line-height: 1.85 !important; }
#qbox textarea {
  font-size: .9375rem !important; font-weight: 600 !important;
  line-height: 1.7 !important;
}
#pbox textarea::placeholder, #qbox textarea::placeholder {
  color: var(--ink-3) !important; font-weight: 400 !important;
}
#pbox textarea:focus, #qbox textarea:focus { outline: none !important; }
#qrow { border-top: 1px solid var(--line-soft); align-items: center !important; }

#send, #send button {
  background: var(--ink) !important; color: var(--canvas) !important;
  border: none !important; border-radius: 8px !important;
  font-size: .8125rem !important; font-weight: 500 !important;
  padding: 8px 16px !important; margin: 4px !important;
  min-width: max-content !important; white-space: nowrap !important;
  flex: 0 0 auto !important; width: auto !important;
}
/* Keep the button's own column from shrinking around it. */
#qrow > div:last-child { flex: 0 0 auto !important; min-width: max-content !important; }
#send:hover, #send button:hover { background: var(--btn-hover) !important; }
#send:focus-visible, #send button:focus-visible { outline: 2px solid var(--mark-edge); outline-offset: 2px; }

#hint {
  margin-top: 8px; font-size: .6875rem; color: var(--ink-3);
  min-height: 1rem; text-align: center;
}

#notfound {
  margin: 14px 32px 0 32px; padding: 11px 14px; border-radius: 7px;
  background: var(--warn-bg); border-left: 3px solid var(--warn-edge);
  color: var(--warn-ink); font-size: .8125rem; line-height: 1.6;
}

/* ---- scrollbar --------------------------------------------------------- */
#rail::-webkit-scrollbar { width: 9px; }
#rail::-webkit-scrollbar-thumb {
  background: var(--scroll); border-radius: 5px; border: 3px solid var(--canvas);
}
#rail::-webkit-scrollbar-thumb:hover { background: var(--scroll-hi); }

@media (max-width: 860px) {
  #shell { flex-direction: column !important; }
  #rail { max-width: 100%; min-width: 0; border-right: none;
          border-bottom: 1px solid var(--line); }
  #card { display: none; }
  #rail { position: static; max-height: none; min-height: 0; }
  #scroll { padding: 20px 18px 16px 18px; }
  #composer { padding: 12px 18px 16px 18px; }
  .turn-n { display: none; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

SCROLL_JS = """
() => {
  requestAnimationFrame(() => {
    const turns = document.querySelectorAll('.turn');
    const last = turns[turns.length - 1];
    if (last) {
      last.scrollIntoView({ block: 'start', behavior: 'smooth' });
    } else {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    }
  });
}
"""

# The button names the mode it switches TO, so it reads as an action.
# Gradio's own .dark class is kept in step so its internals don't disagree
# with ours. localStorage can throw in private windows, hence the try/catch.
THEME_APPLY = """
function themeButton() {
  // Gradio puts elem_id on the <button> itself, but has wrapped it in some
  // versions, so accept either shape.
  const el = document.getElementById('themebtn');
  if (!el) return null;
  return (el.tagName === 'BUTTON') ? el : el.querySelector('button');
}
function applyTheme(t) {
  document.documentElement.setAttribute('data-ui-theme', t);
  const c = document.querySelector('.gradio-container');
  if (c) { c.classList.toggle('dark', t === 'dark'); }
  document.body.classList.toggle('dark', t === 'dark');
  const b = themeButton();
  if (b) { b.textContent = (t === 'dark') ? 'Light mode' : 'Dark mode'; }
}
"""

INIT_JS = """
() => {
  %s
  let t = null;
  try { t = localStorage.getItem('tqa-theme'); } catch (e) {}
  if (t !== 'dark' && t !== 'light') {
    t = window.matchMedia &&
        window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark' : 'light';
  }
  applyTheme(t);
}
""" % THEME_APPLY

TOGGLE_JS = """
() => {
  %s
  const cur = document.documentElement.getAttribute('data-ui-theme');
  const next = (cur === 'dark') ? 'light' : 'dark';
  applyTheme(next);
  try { localStorage.setItem('tqa-theme', next); } catch (e) {}
}
""" % THEME_APPLY


# =============================================================================
# BUILD
# =============================================================================

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui"],
)

with gr.Blocks(title="Telugu math QA", theme=theme, css=CSS,
               analytics_enabled=False) as demo:

    turns_state = gr.State([])

    with gr.Row(elem_id="shell"):

        # ---------------- rail ----------------
        with gr.Column(scale=0, elem_id="rail"):
            gr.HTML(
                '<div id="brand">'
                '<div class="bname">Telugu math QA</div>'
                '<div class="bsub">MuRIL · extractive span</div>'
                '</div>'
            )

            new_btn = gr.Button("New query", elem_id="newq")

            gr.HTML('<div class="rail-h">This session</div>')
            history = gr.Radio(choices=[], label="", show_label=False,
                               container=False, elem_id="history")
            clear_btn = gr.Button("Clear history", elem_id="clearh")

            gr.HTML('<div class="rail-h">Examples</div>')
            examples = gr.Radio(choices=EXAMPLE_LABELS, label="",
                                show_label=False, container=False,
                                elem_id="examples")

            theme_btn = gr.Button("Dark mode", elem_id="themebtn")

            gr.HTML(
                f"""
                <div id="card">
                  <b>Model</b><br>
                  MuRIL base, 237M parameters, fine-tuned for extractive QA<br>
                  <code>{html.escape(os.path.basename(MODEL_PATH))}</code>
                  <code>{DEVICE}</code><br><br>
                  <b>No memory</b> between queries. Each is an independent
                  prediction from the passage you give it.
                </div>
                """
            )

        # ---------------- canvas ----------------
        with gr.Column(scale=1, elem_id="canvas"):

            if not IS_FINETUNED:
                gr.HTML(
                    f'<div id="notfound"><b>Fine-tuned weights not found.</b> '
                    f'Running <code>{FALLBACK_MODEL}</code> with an untrained '
                    f'question-answering head, so its output is noise. Pass '
                    f'<code>--model model/muril_telugu_math_v2</code> to load '
                    f'a trained checkpoint.</div>'
                )

            transcript = gr.HTML(EMPTY_CANVAS, elem_id="scroll")

            with gr.Column(elem_id="composer"):
                with gr.Column(elem_id="cbox"):
                    passage = gr.Textbox(
                        placeholder="Paste a Telugu passage containing the answer",
                        lines=3, max_lines=10, show_label=False,
                        container=False, elem_id="pbox",
                    )
                    with gr.Row(elem_id="qrow"):
                        question = gr.Textbox(
                            placeholder="Ask a question about it",
                            lines=1, max_lines=3, show_label=False,
                            container=False, scale=8, elem_id="qbox",
                        )
                        send = gr.Button("Find answer", scale=0, elem_id="send")
                hint = gr.Markdown("", elem_id="hint")

    # ---------------- wiring ----------------
    submit_args = dict(
        fn=on_submit,
        inputs=[passage, question, turns_state],
        outputs=[transcript, turns_state, history, passage, question, hint],
    )
    send.click(**submit_args).then(fn=None, js=SCROLL_JS)
    question.submit(**submit_args).then(fn=None, js=SCROLL_JS)

    examples.change(
        fn=on_pick_example,
        inputs=[examples, turns_state],
        outputs=[passage, question, hint],
    )
    history.change(
        fn=on_pick_history,
        inputs=[history, turns_state],
        outputs=[passage, question],
    )
    new_btn.click(
        fn=on_new, inputs=None,
        outputs=[passage, question, examples, hint],
    )
    clear_btn.click(
        fn=on_clear_history, inputs=None,
        outputs=[transcript, turns_state, history, hint],
    )
    theme_btn.click(fn=None, inputs=None, outputs=None, js=TOGGLE_JS)

    # Restore the saved theme before first paint, falling back to the OS
    # preference the first time someone opens the app.
    demo.load(fn=None, inputs=None, outputs=None, js=INIT_JS)


if __name__ == "__main__":
    # Gradio 6 moved theme and css from Blocks() to launch(). They are passed
    # in both places so the app styles correctly on 5.x and 6.x alike; older
    # versions reject the launch kwargs, hence the fallback.
    try:
        demo.launch(server_port=ARGS.port, share=ARGS.share, show_error=True,
                    theme=theme, css=CSS)
    except TypeError:
        demo.launch(server_port=ARGS.port, share=ARGS.share, show_error=True)
