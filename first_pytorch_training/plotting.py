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


def plot_lr_experiments(learning_rates, results):
    if not results:
        return

    plt.figure(figsize=(10, 6))
    max_epochs = 0

    for lr, (training_losses, validation_losses) in zip(learning_rates, results):
        epochs = range(1, len(training_losses) + 1)
        max_epochs = max(max_epochs, len(training_losses))

        # Plot training loss with a solid line and capture its color
        line = plt.plot(epochs, training_losses, marker='o', label=f'Train (lr={lr})')
        color = line[0].get_color()

        # Plot validation loss with a dashed line using the exact same color
        plt.plot(epochs, validation_losses, marker='x', linestyle='--', color=color, label=f'Val (lr={lr})')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Across Learning Rates")

    plt.xticks(range(1, max_epochs + 1))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
