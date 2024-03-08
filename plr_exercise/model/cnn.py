from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """Defines a convolutional neural network (CNN) for image classification."""

    def __init__(self):
        """Initializes the CNN layers.

        The network architecture:
            * Two convolutional layers with ReLU activations.
            * Max pooling after the second convolutional layer.
            * Dropout for regularization.
            * Two fully connected layers with ReLU activation on the first.
            * Dropout after the first fully connected layer.
            * Log softmax output layer.
        """

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Defines the forward pass through the network.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The log softmax probabilities for each class.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
