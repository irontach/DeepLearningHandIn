import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

def save_cifar10_numpy_torch():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    print("Downloading CIFAR-10 training data...")
    trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform)
    
    print("Downloading CIFAR-10 test data...")
    testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)


    x_train_tensor = torch.stack([item[0] for item in trainset])
    y_train_tensor = torch.tensor(trainset.targets).reshape(-1, 1)

    x_test_tensor = torch.stack([item[0] for item in testset])
    y_test_tensor = torch.tensor(testset.targets).reshape(-1, 1)
    
    x_train = x_train_tensor.permute(0, 2, 3, 1).numpy()
    x_test = x_test_tensor.permute(0, 2, 3, 1).numpy()
    y_train = y_train_tensor.numpy()
    y_test = y_test_tensor.numpy()
    
    x_train = (x_train * 255).astype(np.uint8)
    x_test = (x_test * 255).astype(np.uint8)
    
    print(f"\nConverted Training data shape: {x_train.shape}")
    
    filename = 'cifar10_torch_data.npz'
    print(f"Saving to {filename}...")

    np.savez_compressed(
        filename,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
    
    print(f"Success! File saved at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    save_cifar10_numpy_torch()