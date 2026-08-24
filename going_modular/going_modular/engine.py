"""
this module contain function for training and testing a pytorch model
"""

from unittest import result

import torch
from tqdm.auto import tqdm
from typing import Dict, List,Tuple

def train_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, optimizer:torch.optim.Optimizer, device:torch.device) -> Tuple[float,float]:
    """
    Performs a single training step for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
        Tuple[float, float]: A tuple containing the average training loss and accuracy for the epoch.
    """
    # Set the model to training mode
    model.train()

    # Initialize variables to track loss and accuracy
    train_loss, train_acc = 0, 0

    # Iterate over batches of data in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Move data to the specified device
        X, y = X.to(device), y.to(device)

        # Forward pass: compute predictions
        y_pred = model(X)

        # Compute the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Zero the gradients before backward pass
        optimizer.zero_grad()

        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate accuracy for this batch
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Calculate average loss and accuracy over all batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc
def test_step(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn:torch.nn.Module, device:torch.device) -> Tuple[float, float]:
    """
    Performs a single testing step for a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be tested.
        dataloader (torch.utils.data.DataLoader): DataLoader for the testing data.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
        Tuple[float, float]: A tuple containing the average testing loss and accuracy for the epoch.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track loss and accuracy
    test_loss, test_acc = 0, 0

    # Iterate over batches of data in the dataloader
    for batch, (X, y) in enumerate(dataloader):
        # Move data to the specified device
        X, y = X.to(device), y.to(device)

        # Forward pass: compute predictions
        with torch.no_grad():
            y_pred = model(X)

        # Compute the loss
        loss = loss_fn(y_pred, y)
        test_loss += loss.item()

        # Calculate accuracy for this batch
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        test_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Calculate average loss and accuracy over all batches
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model:torch.nn.Module, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, loss_fn:torch.nn.Module, epochs:int, device:torch.device) -> Dict[str,List[float]]:
    """
        Trains a PyTorch model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the testing data.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's parameters.
        loss_fn (torch.nn.Module): Loss function to compute the loss.
        epochs (int): Number of epochs to train the model.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
            A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # create empty result 
    result={
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer, device=device)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)

        print(f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}")
            # Append metrics to result dictionary
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["test_loss"].append(test_loss)
        result["test_acc"].append(test_acc)
    return result