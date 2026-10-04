# Superseded artifacts

Kept so the corrections described in the paper can be checked against what
they replaced. None of these are current results.

- `evaluation_results.json`, `evaluation_results_e*.json` — the original
  evaluation run: 90.4 F1 / 88.9 EM on a 190-example test set. That run used a
  randomly initialised baseline, had no early stopping, and its split was not
  leakage-free. The current results are in `../results/multiseed_results.json`.
- `01_gradio_app.py` — the first demo. Its model path points at a folder from
  an earlier layout, so it silently falls back to the untrained base model.
  Use `../chatbot/` instead.
