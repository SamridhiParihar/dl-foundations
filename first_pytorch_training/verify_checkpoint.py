from pathlib import Path
import torch
from torch import nn
from first_pytorch_training.training import train_model
from first_pytorch_training.dataset import train_loader
from first_pytorch_training.dataset import test_loader
from first_pytorch_training.model import MyNeuralNetwork
from first_pytorch_training.checkpoint import load_checkpoint
from first_pytorch_training.checkpoint import save_checkpoint

def checkpoint_verification(model_for_training, filepath, loss_fn, optimizer, device,epoch):

    # Train a model
    train_model(model_for_training, train_loader, loss_fn, optimizer, device, epoch)

    # Get a batch of input images
    X, y = next(iter(test_loader))
    X = X.to(device)

    # Original trained model predictions
    model_for_training.eval()

    with torch.no_grad():
        predictions_trained_model = model_for_training(X)


    # Save checkpoint
    save_checkpoint(model_for_training, filepath)

    # Create a fresh model and load checkpoint
    model = MyNeuralNetwork()
    model = load_checkpoint(model, filepath, device)
    model.eval()

    # Predictions from restored model
    with torch.no_grad():
        predictions = model(X)

    # Verify numerical equivalence of logits
    torch.testing.assert_close(
        predictions_trained_model,
        predictions,
        rtol=1e-5,
        atol=1e-6
    )    
    # Compare predicted classes
    original_classes = predictions_trained_model.argmax(dim=1)
    restored_classes = predictions.argmax(dim=1)

    same_predictions = (original_classes == restored_classes).sum().item()

    total_samples = y.size(0)
    agreement_percentage = (same_predictions / total_samples) * 100

    return agreement_percentage

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model_for_training = MyNeuralNetwork()
    model_for_training.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_for_training.parameters(), lr=1e-3)
    filepath = Path(__file__).resolve().parent / "checkpoints" / "fashion_mnist2.pth"
    epoch = 5

    print(checkpoint_verification(model_for_training,filepath,loss_fn,optimizer,device,epoch))





