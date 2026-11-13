import matplotlib.pyplot as plt 

import matplotlib.pyplot as plt


def plot_losses(training_losses, validation_losses):

    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(8, 5))

    # Plot training loss
    plt.plot(epochs, training_losses, marker='o', label='Training Loss')

    # Plot validation loss
    plt.plot(epochs, validation_losses, marker='o', label='Validation Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
