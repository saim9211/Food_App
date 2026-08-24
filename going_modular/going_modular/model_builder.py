"""
Contains pytorch model code to instantiate a tiny model for the pizza, steak, sushi dataset."""
import torch
from torch import nn

class TinnyGG(nn.Module):
    """
    A tiny model for the pizza, steak, sushi dataset.
    """

    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        """
        Initializes the TinnyGG model.

        Args:
            input_shape (int): The number of input features.
            hidden_units (int): The number of hidden units in the hidden layer.
            output_shape (int): The number of output classes.
        """
        super().__init__()
        self.layer_stack1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=0,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer_stack2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 15 * 15, out_features=output_shape)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the TinnyGG model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.layer_stack1(x)
        x = self.layer_stack2(x)
        x = self.classifier(x)
        return x