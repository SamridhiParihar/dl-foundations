# ============================================================
# Q5. SAVE YOUR VISUALIZATION
# ============================================================
#
# Using the training and validation loss data from Q4:
#
# Tasks:
#
# 1. Create an outputs directory using pathlib.Path.
#
# 2. Create the Figure and Axes.
#
# 3. Plot both training and validation loss.
#
# 4. Apply tight_layout().
#
# 5. Save the Figure as:
#
#    outputs/training_vs_validation.png
#
# 6. Use dpi=150.
#
# 7. Use bbox_inches="tight".
#
# 8. Display the plot.
#
# 9. Close the Figure after displaying it.
#
# 10. Verify that the image file exists.
#
# ============================================================

from pathlib import Path
import matplotlib.pyplot as plt

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
path = output_dir / "training_vs_validation.png"

epochs = [1, 2, 3, 4, 5, 6]
train_loss = [2.5, 1.8, 1.2, 0.8, 0.5, 0.3]
val_loss = [2.7, 2.0, 1.5, 1.1, 0.9, 0.8]

fig, axes = plt.subplots()

axes.plot(epochs, train_loss, marker="o", label="Training Loss")
axes.plot(epochs, val_loss, marker="s", linestyle="--", label="Validation Loss")

axes.set_title("Training vs. Validation Loss")
axes.set_xlabel("Epochs")
axes.set_ylabel("Loss")
axes.grid(True, alpha=0.6)
axes.legend()

fig.tight_layout()

fig.savefig(path, dpi=150, bbox_inches="tight")

plt.show()

plt.close(fig)

if path.exists():
  print(f"File successfully saved at: {path.resolve()}")
else:
  print("Error: File was not found.")


