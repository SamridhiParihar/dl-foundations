# ============================================================
# Q2. CUSTOMIZE A LINE PLOT
# ============================================================
#
# Given:
#
# epochs = [1, 2, 3, 4, 5, 6]
# accuracy = [45, 55, 63, 72, 81, 88]
#
# Tasks:
#
# 1. Create a Figure with figsize=(8, 5).
#
# 2. Plot the accuracy against epochs.
#
# 3. Use a green line.
#
# 4. Use circular markers at every data point.
#
# 5. Add a label "Training Accuracy".
#
# 6. Set appropriate title and axis labels.
#
# 7. Display a legend.
#
# 8. Enable grid lines.
#
# 9. Display the plot.
#
# ============================================================

import matplotlib.pyplot as plt 

epochs = [1, 2, 3, 4, 5,6]
accuracy = [45, 55, 63, 72, 81, 88]

fig, axes = plt.subplots(figsize=(8,5))

axes.plot(epochs,accuracy , color="green", marker='o',label='Training Accuracy',linestyle="--")
axes.set_title("Training Accuracy Over Epochs")
axes.set_xlabel("Epoch")
axes.set_ylabel("Accuracy")
axes.legend()
axes.grid(True,linestyle='--',alpha=0.6)
plt.show()
