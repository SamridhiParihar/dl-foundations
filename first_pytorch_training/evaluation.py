import torch
from torch import nn


def evaluate_model(model,data_loader,loss_fn,device):

    # setting to evaluation mode
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for X,y in data_loader : 
        
                #  Move images and labels to device.
                X = X.to(device)
                y = y.to(device)
        
                # forward pass 
                prediction = model(X)
        
                # loss calculation 
                loss = loss_fn(prediction,y) 

                # predicted classes 
                predicted_classes = prediction.argmax(dim=1)

                # counting correct prediciton 
                correct_predictions += (predicted_classes == y).sum().item()

                total_loss+= loss.item()*X.size(0)
                total_samples+=X.size(0)
            
    # Calculate average loss and accuracy.
    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100

    return avg_loss, accuracy


