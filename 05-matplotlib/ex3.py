# ============================================================
# Q3. CREATE MULTIPLE SUBPLOTS
# ============================================================
#
# Given:
#
# epochs = [1, 2, 3, 4, 5]
#
# train_loss = [2.5, 1.8, 1.2, 0.8, 0.5]
# train_accuracy = [45, 55, 65, 75, 85]
#
# Tasks:
#
# 1. Create one Figure containing two Axes.
#
# 2. Arrange the Axes horizontally (1 row, 2 columns).
#
# 3. Plot training loss on the first Axes.
#
# 4. Plot training accuracy on the second Axes.
#
# 5. Give each graph an appropriate title.
#
# 6. Label both X and Y axes.
#
# 7. Use different markers for the two graphs.
#
# 8. Apply tight_layout().
#
# 9. Display the complete Figure.
#
# ============================================================
import matplotlib.pyplot as plt 

epochs = [1, 2, 3, 4, 5]
loss = [2.5, 1.8, 1.2, 0.8, 0.5]
accuracy = [45, 55, 63, 72, 81]

fig , ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

ax[0].plot(epochs,loss,marker='o')
ax[0].set_title("Training Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].grid(True,linestyle='--',alpha=0.6)


ax[1].plot(epochs,accuracy,marker='s')
ax[1].set_title("Training Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].grid(True,linestyle='--',alpha=0.6)

fig.tight_layout()
plt.show()

