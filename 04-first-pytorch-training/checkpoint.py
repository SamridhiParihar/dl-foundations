import torch 

# saving the trained model's learned parameters to a file.
def save_checkpoint(model,filepath):
    state_dict = model.state_dict()
    torch.save(state_dict,filepath)

# loading the trained model's learned parameters 

def load_checkpoint(model,filepath,device):

    state_dict = torch.load(filepath,map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    return model

