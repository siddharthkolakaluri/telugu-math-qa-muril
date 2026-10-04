import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from collections import defaultdict

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT  = _SCRIPT_DIR   # script lives in repo root
except NameError:
    # Running inside a notebook cell
    _REPO_ROOT = os.path.expanduser(
        "Users/siddharth/Documents/nlp_project"
    )

LOG_PATH   = os.path.join(_REPO_ROOT, "model", "training_log.json")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "figures")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fig3_loss_curves.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


C_BLUE   = "#2E5FA3"   # training loss line
C_ORANGE = "#E07B39"   # validation loss line
C_GREEN  = "#3A9E5F"   # best val loss marker
C_DARK   = "#1F2937"   # text


plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.9,
})

print(f" Loading training log from: {LOG_PATH}")

with open(LOG_PATH, encoding="utf-8") as f:
    log = json.load(f)

# Separate training step entries from evaluation entries
# Training entries have key "loss" but NOT "eval_loss"
# Evaluation entries have key "eval_loss"
train_entries = [e for e in log if "loss" in e and "eval_loss" not in e]
eval_entries  = [e for e in log if "eval_loss" in e]

print(f"   Train step entries : {len(train_entries)}")
print(f"   Eval entries       : {len(eval_entries)}")

# ── Compute per-epoch average training loss ───────────────────────────────────
# The log records training loss every 10 steps (38 entries per epoch).
# We group by integer epoch and average to get one value per epoch.
epoch_train = defaultdict(list)
for entry in train_entries:
    ep = int(entry["epoch"])
    ep = max(ep, 1)   # treat sub-epoch 0 entries as epoch 1
    epoch_train[ep].append(entry["loss"])

train_epochs   = sorted(epoch_train.keys())
train_loss_avg = [np.mean(epoch_train[ep]) for ep in train_epochs]

# ── Validation loss — one entry per epoch ────────────────────────────────────
eval_epochs    = [e["epoch"]    for e in eval_entries]
eval_loss_vals = [e["eval_loss"] for e in eval_entries]

# ── Find best validation checkpoint ──────────────────────────────────────────
best_val_loss  = min(eval_loss_vals)
best_val_epoch = eval_epochs[eval_loss_vals.index(best_val_loss)]

print(f"\n   Initial train loss : {train_loss_avg[0]:.4f}")
print(f"   Final train loss   : {train_loss_avg[-1]:.6f}")
print(f"   Initial val loss   : {eval_loss_vals[0]:.4f}")
print(f"   Final val loss     : {eval_loss_vals[-1]:.4f}")
print(f"   Best val loss      : {best_val_loss:.4f}  (epoch {int(best_val_epoch)})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── LEFT PANEL: Linear scale
ax1 = axes[0]

ax1.plot(train_epochs, train_loss_avg,
         color=C_BLUE, lw=2.0, label="Training Loss", zorder=3)

ax1.plot(eval_epochs, eval_loss_vals,
         color=C_ORANGE, lw=2.0, linestyle="--",
         label="Validation Loss", zorder=3,
         marker="o", markersize=2.5, markevery=5)

# Annotate starting point
ax1.annotate(
    f"Epoch 1\nTrain: {train_loss_avg[0]:.2f}",
    xy=(1, train_loss_avg[0]),
    xytext=(8, train_loss_avg[0] - 0.4),
    fontsize=7.5, color=C_BLUE,
    arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1)
)

# Annotate final point
ax1.annotate(
    f"Epoch 100\nTrain: {train_loss_avg[-1]:.4f}\nVal: {eval_loss_vals[-1]:.3f}",
    xy=(100, eval_loss_vals[-1]),
    xytext=(72, eval_loss_vals[-1] + 0.8),
    fontsize=7.5, color=C_ORANGE,
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1)
)

# Mark best validation checkpoint
ax1.axvline(best_val_epoch, color=C_GREEN, lw=1.2, linestyle=":",
            label=f"Best val checkpoint (epoch {int(best_val_epoch)})", zorder=2)
ax1.plot(best_val_epoch, best_val_loss, "o", color=C_GREEN, markersize=8, zorder=4)
ax1.text(best_val_epoch + 1.5, best_val_loss + 0.05,
         f"Best val\n{best_val_loss:.3f}", fontsize=7.5, color=C_GREEN)

ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
ax1.set_title("Linear Scale", fontsize=12, pad=8)
ax1.legend(loc="upper right")
ax1.set_xlim(0, 102)
ax1.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax1.set_axisbelow(True)
ax1.spines["left"].set_visible(True)
ax1.spines["bottom"].set_visible(True)

# ── RIGHT PANEL: Log scale ────────────────────────────────────────────────────
ax2 = axes[1]

ax2.semilogy(train_epochs, train_loss_avg,
             color=C_BLUE, lw=2.0, label="Training Loss", zorder=3)

ax2.semilogy(eval_epochs, eval_loss_vals,
             color=C_ORANGE, lw=2.0, linestyle="--",
             label="Validation Loss", zorder=3,
             marker="o", markersize=2.5, markevery=5)

ax2.axvline(best_val_epoch, color=C_GREEN, lw=1.2, linestyle=":",
            label=f"Best val checkpoint (epoch {int(best_val_epoch)})", zorder=2)

# Label final values on the right edge
ax2.text(101, train_loss_avg[-1],
         f" {train_loss_avg[-1]:.2e}", fontsize=7.5, color=C_BLUE, va="center")
ax2.text(101, eval_loss_vals[-1],
         f" {eval_loss_vals[-1]:.3f}", fontsize=7.5, color=C_ORANGE, va="center")

ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Loss (log scale)", fontsize=11)
ax2.set_title("Log Scale", fontsize=12, pad=8)
ax2.legend(loc="upper right")
ax2.set_xlim(0, 102)
ax2.yaxis.grid(True, which="both", linestyle="--", alpha=0.3, zorder=0)
ax2.set_axisbelow(True)
ax2.spines["left"].set_visible(True)
ax2.spines["bottom"].set_visible(True)

# =============================================================================
# SAVE
# =============================================================================

plt.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\n Figure saved to: {OUTPUT_FILE}")
print(f"\n   Paper caption:")
print(f"   Figure 3: Training and validation loss over 100 epochs.")
print(f"   (Left) Linear scale. Training loss fell from {train_loss_avg[0]:.2f} to")
print(f"   effectively zero while validation loss reached a minimum of")
print(f"   {best_val_loss:.3f} at epoch {int(best_val_epoch)} before rising (overfitting).")
print(f"   (Right) Log scale showing the same curves. The green dotted line")
print(f"   marks the best model checkpoint retained for evaluation.")
