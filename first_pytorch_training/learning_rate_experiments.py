import torch 
from torch import nn
from first_pytorch_training.model import MyNeuralNetwork
from first_pytorch_training.training import train_model_with_val
from first_pytorch_training.dataset import train_loader
from first_pytorch_training.dataset import val_loader
from first_pytorch_training.plotting import plot_lr_experiments


def learning_rate_experiments(learning_rates, device, epochs, train_loader, val_loader):
    results = []

    for lr in learning_rates:
        print(f"start training for learning rate : {lr}")
        # Reset random seed so every model starts with identical weights
        torch.manual_seed(42)

        # Reset batch ordering
        train_loader.generator.manual_seed(42)

        # Create a fresh model
        model = MyNeuralNetwork().to(device)

        # Create a fresh optimizer for this learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create a fresh loss function
        loss_fn = nn.CrossEntropyLoss()

        # Train and validate
        epoch_losses_train, epoch_losses_val = train_model_with_val(
            model, train_loader, val_loader, loss_fn, optimizer, device, epochs
        )

        results.append((epoch_losses_train, epoch_losses_val))

        print(f"ends learning for learning rate : {lr}")
        print()

    return results




if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    learning_rates = [1e-1,1e-2,1e-3]
    epochs= 5

    results = learning_rate_experiments(learning_rates,device,epochs,train_loader,val_loader)
    plot_lr_experiments(learning_rates,results)

