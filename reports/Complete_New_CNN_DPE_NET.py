# python -m reports.Complete_New_CNN_ensemble


import matplotlib.pyplot as plt
import numpy as np

# --- Data from your logs ---
epochs = list(range(1, 47))

train_loss = [
0.081872,0.051811,0.042293,0.042836,0.039830,0.035502,0.032924,0.033526,0.032273,0.031172,
0.022102,0.028429,0.025982,0.030906,0.023195,0.024352,0.025869,0.022792,0.021533,0.018833,
0.019393,0.018406,0.015721,0.020618,0.022305,0.018717,0.018608,0.016355,0.014858,0.016425,
0.015927,0.016120,0.016323,0.015016,0.016068,0.015867,0.016082,0.016167,0.015889,0.014401,
0.014659,0.015718,0.015904,0.014985,0.013400,0.013290
]

val_loss = [
0.128431,0.153375,0.080618,0.244203,0.068704,0.073468,0.083919,0.105570,0.103907,0.065901,
0.063552,0.110511,0.122773,0.076781,0.057613,0.074149,0.065814,0.062939,0.090532,0.053463,
0.064508,0.059087,0.060021,0.049526,0.052147,0.056387,0.059987,0.050501,0.051768,0.047837,
0.050134,0.045163,0.045101,0.046078,0.046336,0.041622,0.050306,0.044370,0.044264,0.054407,
0.045409,0.043158,0.045272,0.046623,0.044141,0.043552
]

# --- Best epoch ---
best_epoch = 36
best_val = 0.041622

# --- Plot ---
plt.figure(figsize=(12,6))

plt.plot(epochs, train_loss, 'o-', label='Training Loss', color='blue')
plt.plot(epochs, val_loss, 's-', label='Validation Loss', color='red')

# Best point
plt.scatter(best_epoch, best_val, color='gold', s=120, zorder=5, label='Best Weights Saved')

# Vertical line
plt.axvline(best_epoch, linestyle='--', color='green', alpha=0.7)

# Annotation
plt.annotate(
    f'Best Model\nEpoch: {best_epoch}\nVal Loss: {best_val:.4f}',
    xy=(best_epoch, best_val),
    xytext=(best_epoch-12, best_val+0.08),
    arrowprops=dict(facecolor='black', arrowstyle='->'),
    bbox=dict(
        boxstyle="round,pad=0.3",
        edgecolor='green',
        facecolor='none'   # 👈 makes it transparent
    )
)

# Titles
plt.title("Training and Validation Loss vs Epoch\nDPE-Net (Dual-Path Patch Encoder)", fontsize=16)
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Extra info box
info_text = (
    "Writers: Train=297 | Val=74\n"
    "Total Writers: 371\n"
    "Params: 0.1775M\n"
    "FLOPs: 58.49M\n"
    "Avg Epoch Time: 17.06 min"
)

plt.gcf().text(0.72, 0.55, info_text,
               fontsize=10,
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()