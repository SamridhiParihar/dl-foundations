import torch
from torch import nn

from model import MyNeuralNetwork


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


def train_one_epoch(model,train_loader,loss_fn,optimizer,device):
    '''model : the model we select for training 
       train_loader : the datasetloader for shuffled data from train_dataset
       loss_fn : to calculate loss 
       optimizer : for gradeint descent step and resetting the grad 
       device : where we want to the actual mathematical operatio to happen : 'mps' or 'cpu' 
    '''
    # Define the function and its parameters.
    # Set model.train().
    # Iterate through train_loader.
    # Move images and labels to DEVICE.
    # Implement forward pass and loss calculation.
    # Implement gradient reset, backpropagation, and weight updates.
    # Return average training loss.

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



if __name__ == "__main__":

    from dataset import train_loader

    # epoch_loss = train_one_epoch(
    #     model, train_loader, loss_fn, optimizer, DEVICE
    # )

    epoch_losses = train_model(
            model, train_loader, loss_fn, optimizer, DEVICE , 5
        )

    print("Training loss:", epoch_losses)

