# phase 1 : --------------------- data pipeline ---------------------------------------
# imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms , datasets
from torch.utils.data import random_split


# constants 
BATCH_SIZE = 64
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 1. Define transforms (converts PIL images to normalized float tensors)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

# 2. Download and load the full training set and test set
full_train_dataset = datasets.FashionMNIST (
    root='data',
    train=True,
    download=True,
    transform=transform 
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=transform    
)

# 3. Create an 90/10 Train / Validation split (48,000 train, 12,000 val)
full_train_size = len(full_train_dataset)
train_size = int(0.9*full_train_size)
val_size = full_train_size - train_size

# Use a fixed generator seed for reproducible splits
generator = torch.Generator()
generator.manual_seed(42)
train_dataset , val_dataset = random_split(
    full_train_dataset,
    [train_size,val_size],
    generator = generator
)


train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 0,
    pin_memory = False # for mac 
)

val_loader = DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 0,
    pin_memory = False # for mac 
)

test_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False,
    num_workers = 0,
    pin_memory = False # for mac 
)




# Verification of data pipeline
if __name__ == "__main__":

    print(
        f"Dataset sizes:\n"
        f"Train: {len(train_dataset)}\n"
        f"Validation: {len(val_dataset)}\n"
        f"Test: {len(test_dataset)}"
    )

    images, labels = next(iter(train_loader))

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")
    print(f"Images min: {images.min()}")
    print(f"Images max: {images.max()}")