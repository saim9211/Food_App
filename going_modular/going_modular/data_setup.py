"""
This is a function for creating pytorch dataloaders for the pizza, steak, sushi dataset.
"""

import os
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

Num_workers = 0

def create_dataloaders(train_dir:str, test_dir:str, transform:transforms.Compose, batch_size:int):
    """
    Creates training and test dataloaders for the pizza, steak, sushi dataset.

    Args:
        train_dir (str): Directory path to the training data.
        test_dir (str): Directory path to the test data.
        transform (transforms.Compose): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.
    Returns:
        train_dataloader (DataLoader): DataLoader for the training data.
        test_dataloader (DataLoader): DataLoader for the test data.
    """

    # Create training and test datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)
    # create the class name
    class_names = train_data.classes
    # Create training and test dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=Num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=Num_workers)

    return train_dataloader, test_dataloader, class_names