
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import Module

from model import MySimpleCNN

from first_pytorch_training.training import train_model_with_val
from first_pytorch_training.evaluation import evaluate_model
from first_pytorch_training.dataset import (
    train_loader,
    val_loader,
    test_loader,
)


# ============================================================
# MLP MODEL
# Architecture: 784 -> 128 -> 64 -> 10
# ============================================================

class MyMLP(Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()


def train_and_evaluate(model, model_name, device, epochs=5):

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    print(f"\n{'=' * 55}")
    print(f"Training {model_name}")
    print(f"{'=' * 55}")

    parameters = count_parameters(model)

    print(f"Trainable parameters: {parameters:,}")

    synchronize_device(device)
    start_time = time.perf_counter()

    train_losses, val_losses = train_model_with_val(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        epochs
    )

    synchronize_device(device)
    training_time = time.perf_counter() - start_time

    val_loss, val_accuracy = evaluate_model(
        model,
        val_loader,
        loss_fn,
        device
    )

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        loss_fn,
        device
    )

    print(f"\n{model_name} Results")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2f}%")

    return {
        "name": model_name,
        "parameters": parameters,
        "training_time": training_time,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    EPOCHS = 5

    print(f"Using device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")

    # --------------------------------------------------------
    # 1. Train MLP
    # --------------------------------------------------------

    torch.manual_seed(42)

    mlp = MyMLP()

    mlp_results = train_and_evaluate(
        mlp,
        "MLP",
        DEVICE,
        EPOCHS
    )

    # --------------------------------------------------------
    # 2. Train CNN
    # --------------------------------------------------------

    torch.manual_seed(42)

    cnn = MySimpleCNN()

    cnn_results = train_and_evaluate(
        cnn,
        "CNN",
        DEVICE,
        EPOCHS
    )

    # --------------------------------------------------------
    # 3. Compare Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("                  MLP vs CNN COMPARISON")
    print("=" * 75)

    print(
        f"{'Metric':<25}"
        f"{'MLP':>20}"
        f"{'CNN':>20}"
    )

    print("-" * 75)

    metrics = [
        ("Parameters", "parameters", ","),
        ("Training time (sec)", "training_time", ".2f"),
        ("Validation loss", "val_loss", ".4f"),
        ("Validation accuracy (%)", "val_accuracy", ".2f"),
        ("Test loss", "test_loss", ".4f"),
        ("Test accuracy (%)", "test_accuracy", ".2f"),
    ]

    for label, key, fmt in metrics:

        mlp_value = mlp_results[key]
        cnn_value = cnn_results[key]

        if fmt == ",":
            mlp_display = f"{mlp_value:,}"
            cnn_display = f"{cnn_value:,}"
        else:
            mlp_display = format(mlp_value, fmt)
            cnn_display = format(cnn_value, fmt)

        print(
            f"{label:<25}"
            f"{mlp_display:>20}"
            f"{cnn_display:>20}"
        )

    print("=" * 75)

    # --------------------------------------------------------
    # 4. Plot Loss Comparison
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    epochs_range = range(1, EPOCHS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training loss
    axes[0].plot(
        epochs_range,
        mlp_results["train_losses"],
        marker="o",
        label="MLP"
    )

    axes[0].plot(
        epochs_range,
        cnn_results["train_losses"],
        marker="o",
        label="CNN"
    )

    axes[0].set_title("Training Loss Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Validation loss
    axes[1].plot(
        epochs_range,
        mlp_results["val_losses"],
        marker="o",
        label="MLP"
    )

    axes[1].plot(
        epochs_range,
        cnn_results["val_losses"],
        marker="o",
        label="CNN"
    )

    axes[1].set_title("Validation Loss Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / "mlp_vs_cnn_loss.png"

    fig.savefig(plot_path)
    print(f"\nLoss comparison saved to: {plot_path}")

    plt.show()
