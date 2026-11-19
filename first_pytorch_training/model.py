import torch
from torch import nn


class MyNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# smaller test version before actual training 

if __name__ == "__main__":
    from first_pytorch_training.dataset import train_loader

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MyNeuralNetwork().to(   
        device
    )

    images,label = next(iter(train_loader))
    images = images.to(device)

    outputs = model(images)
    print(len(train_loader))
    print("Input shape:", images.shape)
    print("Output shape:", outputs.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))