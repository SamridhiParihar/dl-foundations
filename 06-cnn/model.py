import torch 
from torch import nn
from torch.nn import Module 
from torchvision import datasets

class MySimpleCNN(Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1 
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1 
        )
        self.relu = nn.ReLU()


        self.max_pool_2d = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Linear(32*14*14,10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.max_pool_2d(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


if __name__ == "__main__":
    model = MySimpleCNN()

    dummy_input = torch.randn(64, 1, 28, 28)

    output = model(dummy_input)

    print(output.shape)


