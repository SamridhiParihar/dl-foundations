
from pathlib import Path
import torch
from torch import nn

from model import MySimpleCNN

from first_pytorch_training.training import train_model_with_val
from first_pytorch_training.plotting import plot_losses
from first_pytorch_training.checkpoint import save_checkpoint
from first_pytorch_training.evaluation import evaluate_model
from first_pytorch_training.dataset import train_loader, val_loader, test_loader


if __name__ == "__main__":

    DEVICE = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    model = MySimpleCNN().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    epoch_losses_train, epoch_losses_val = train_model_with_val(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        DEVICE,
        epochs=5
    )

    # Validation evaluation
    val_loss, val_accuracy = evaluate_model(
        model,
        val_loader,
        loss_fn,
        DEVICE
    )

    print("\nFinal Validation Results")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save checkpoint
    save_checkpoint(
        model,
        Path(__file__).resolve().parent
        / "checkpoints"
        / "cnn_fashion_mnist.pth"
    )

    # Final test evaluation
    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        loss_fn,
        DEVICE
    )

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Plot training vs validation loss
    plot_losses(epoch_losses_train, epoch_losses_val)
