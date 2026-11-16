# ============================================================
# Q4. VISUALIZE OVERFITTING
# ============================================================
#
# Given:
#
# epochs = [1, 2, 3, 4, 5, 6, 7, 8]
#
# train_loss = [
#     2.5, 1.8, 1.2, 0.8,
#     0.5, 0.3, 0.2, 0.1
# ]
#
# val_loss = [
#     2.7, 2.0, 1.5, 1.2,
#     1.1, 1.2, 1.5, 1.9
# ]
#
# Tasks:
#
# 1. Create one Figure and one Axes.
#
# 2. Plot training loss.
#
# 3. Plot validation loss on the SAME Axes.
#
# 4. Use different colors and markers.
#
# 5. Add labels to both curves.
#
# 6. Add a legend.
#
# 7. Add a grid.
#
# 8. Give the graph an appropriate title.
#
# 9. Identify the epoch at which validation loss
#    reaches its minimum.
#
# 10. Explain what happens after that epoch.
#
# ============================================================

import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6, 7, 8]

train_loss = [2.5, 1.8, 1.3, 0.9, 0.6, 0.4, 0.3, 0.2]

val_loss = [2.7, 2.0, 1.5, 1.1, 0.9, 0.85, 0.95, 1.2]

fig,axes = plt.subplots()

axes.plot(epochs,train_loss,marker='o',label='Training Loss')
axes.plot(epochs,val_loss,marker='s',label='Validation Loss')
axes.set_title('Epoch vs Training Loss vs Validation Loss')
axes.set_xlabel('Epoch')
axes.set_ylabel('Loss')
axes.legend()
axes.grid(True,linestyle='--',alpha=0.6)

# Mark the minimum validation loss
min_index = val_loss.index(min(val_loss))

axes.scatter(
    epochs[min_index],
    val_loss[min_index],
    color='black',
    zorder=5
)

axes.annotate(
    f'Minimum: {val_loss[min_index]}',
    xy=(epochs[min_index], val_loss[min_index]),
    xytext=(epochs[min_index] + 0.5, val_loss[min_index] - 0.3),
    arrowprops=dict(arrowstyle='->')
)

plt.show()