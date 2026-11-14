import torch
from torch import nn
from evaluation import evaluate_model
from model import MyNeuralNetwork
from dataset import train_loader , val_loader
from plotting import plot_losses
from checkpoint import save_checkpoint




def train_one_epoch(model,train_loader,loss_fn,optimizer,device):
    '''model : the model we select for training 
       train_loader : the datasetloader for shuffled data from train_dataset
       loss_fn : to calculate loss 
       optimizer : for gradeint descent step and resetting the grad 
       device : where we want to the actual mathematical operatio to happen : 'mps' or 'cpu' 
    '''


    model.train()

    total_loss = 0.0
    total_samples = 0

    for X,y in train_loader : 
        # move the data to device where model is 
        X = X.to(device)
        y = y.to(device)

        # forward pass 
        predictions = model(X)

        # loss calculation
        loss = loss_fn(predictions,y)

        # clearing prev gradients
        optimizer.zero_grad()

        # calculates grads via back prop
        loss.backward()

        # update the parameters for model
        optimizer.step()

        total_loss+= loss.item()*X.size(0)
        total_samples+=X.size(0)
    
    avg_training_loss = total_loss / total_samples

    return avg_training_loss


def train_model(model,train_loader,loss_fn,optimizer,device,epochs):

    epoch_losses = []

    for i in range(epochs):
        epoch_loss = train_one_epoch(model,train_loader,loss_fn,optimizer,device)
        epoch_losses.append(epoch_loss)

        print(f"epoch no : [{i+1}/{epochs}] - Training loss : {epoch_loss}")

    return epoch_losses



# modified train_model to store val losses also 
def train_model_with_val(model,train_loader,val_loader , loss_fn,optimizer,device,epochs):

    epoch_losses_train = []
    epoch_losses_val = []

    for i in range(epochs):
        epoch_loss_train = train_one_epoch(model,train_loader,loss_fn,optimizer,device)
        epoch_loss_val, _ = evaluate_model(model,val_loader,loss_fn,device)
        epoch_losses_train.append(epoch_loss_train)
        epoch_losses_val.append(epoch_loss_val)

        print(
            f"Epoch [{i+1}/{epochs}] - "
            f"Training Loss: {epoch_loss_train:.4f} - "
            f"Validation Loss: {epoch_loss_val:.4f}"
        )
    return epoch_losses_train, epoch_losses_val


if __name__ == "__main__":



    # before training loop we need : model , loss_fn, optimizer , device should be same 

    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    model = MyNeuralNetwork().to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # verification prints 

    print("Device:", DEVICE)
    print("Loss function:", loss_fn)
    print("Optimizer:", optimizer)
    print("Learning rate:", optimizer.param_groups[0]["lr"])

    
    # # training the model 
    # epoch_losses_train = train_model(
    #         model, train_loader ,loss_fn, optimizer, DEVICE , 5
    #     )

    # print("Training loss:", epoch_losses_train)

    # # evaluating the model 
    # avg_loss , accuracy = evaluate_model(model,val_loader,loss_fn,DEVICE)

    # print(f"Average Validation Loss: {avg_loss:.4f}")
    # print(f"Validation Accuracy: {accuracy:.2f}%")


    # training the model with new method (train_model_with_val)
    epoch_losses_train, epoch_losses_val = train_model_with_val(
                model, train_loader,val_loader ,loss_fn, optimizer, DEVICE , 5
    )

    # saving the check point 
    save_checkpoint(model,"checkpoints/fashion_mnist.pth")

    # plotting 
    plot_losses(epoch_losses_train, epoch_losses_val)