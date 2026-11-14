import torch 
from torch import nn
from model import MyNeuralNetwork
from evaluation import evaluate_model
from checkpoint import load_checkpoint
from dataset import test_loader

def test_evaluation(model,device,filepath,loss_fn,data_loader):

    # loading the saved checkpoint 
    model = load_checkpoint(model,filepath,device)
    avg_loss, accuracy = evaluate_model(model,data_loader,loss_fn,device)

    return avg_loss, accuracy

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = MyNeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    filepath = "checkpoints/fashion_mnist2.pth"
    print(test_evaluation(model,device,filepath, loss_fn, test_loader))


