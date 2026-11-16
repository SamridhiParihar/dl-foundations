# ============================================================
# Q6. BUILD A COMPLETE EXPERIMENT VISUALIZATION
# ============================================================
#
# Given:
#
# epochs = [1, 2, 3, 4, 5, 6, 7, 8]
#
# train_loss = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1]
# val_loss   = [2.7, 2.0, 1.5, 1.2, 1.1, 1.2, 1.5, 1.9]
#
# train_accuracy = [45, 55, 65, 72, 80, 85, 89, 93]
# val_accuracy   = [43, 53, 63, 70, 76, 75, 71, 67]
#
# Tasks:
#
# 1. Create one Figure with two vertically arranged Axes.
#
# 2. First Axes:
#    Plot training loss and validation loss.
#
# 3. Second Axes:
#    Plot training accuracy and validation accuracy.
#
# 4. Add titles, labels, legends and grids.
#
# 5. Apply tight_layout().
#
# 6. Save the Figure as:
#    outputs/complete_training_report.png
#
# 7. Interpret both graphs:
#    - Is the model learning?
#    - Is the model overfitting?
#    - At which epoch does validation loss reach its minimum?
#    - What happens to validation accuracy afterward?
#
# 8. Close the Figure.
#
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8]

train_loss = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1]
val_loss = [2.7, 2.0, 1.5, 1.2, 1.1, 1.2, 1.5, 1.9]

train_accuracy = [45, 55, 65, 72, 80, 85, 89, 93]
val_accuracy = [43, 53, 63, 70, 76, 75, 71, 67]

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
save_path = output_dir / "complete_training_report.png"

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

ax[0].plot(epochs, train_loss, marker="o", color="blue", label="Training Loss")
ax[0].plot(
    epochs,
    val_loss,
    marker="s",
    linestyle="--",
    color="orange",
    label="Validation Loss",
)
ax[0].set_title("Training vs. Validation Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].grid(True, linestyle="--", alpha=0.6)
ax[0].legend()

ax[1].plot(
    epochs, train_accuracy, marker="o", color="green", label="Training Accuracy"
)
ax[1].plot(
    epochs,
    val_accuracy,
    marker="s",
    linestyle="--",
    color="red",
    label="Validation Accuracy",
)
ax[1].set_title("Training vs. Validation Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy (%)")
ax[1].grid(True, linestyle="--", alpha=0.6)
ax[1].legend()

fig.tight_layout()

fig.savefig(save_path, dpi=150, bbox_inches="tight")

plt.show()

plt.close(fig)

if save_path.exists():
  print(f"Report saved successfully at: {save_path.resolve()}")
