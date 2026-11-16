# ============================================================
# Q1. CREATE YOUR FIRST PLOT
# ============================================================
#
# Given:
#
# epochs = [1, 2, 3, 4, 5]
# loss = [2.5, 1.8, 1.2, 0.8, 0.5]
#
# Tasks:
#
# 1. Create a Figure and one Axes using plt.subplots().
#
# 2. Plot epochs on the X-axis and loss on the Y-axis.
#
# 3. Set the title to "Training Loss".
#
# 4. Label the X-axis as "Epoch".
#
# 5. Label the Y-axis as "Loss".
#
# 6. Display the plot.
#
# 7. Print the types of fig and ax.
#
# Expected understanding:
# - What is a Figure?
# - What is an Axes?
# - What does ax.plot() do?
#
# ============================================================

import matplotlib.pyplot as plt 

epochs = [1, 2, 3, 4, 5]
loss = [2.5, 1.8, 1.2, 0.8, 0.5]

fig, axes = plt.subplots()

axes.plot(epochs,loss)
axes.set_title("Training Loss")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
plt.show()
print(f"fig : {type(fig)} ...axes : {type(axes)}")
