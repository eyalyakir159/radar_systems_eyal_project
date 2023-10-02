import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearNetwork(nn.Module):
    def __init__(self,input_size,output_size):
        super(SimpleLinearNetwork, self).__init__()


        # Define the layers
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer with 128 units
        self.dropout1 = nn.Dropout(0.3)  # Dropout layer with 50% dropout rate after fc1
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 units
        self.dropout2 = nn.Dropout(0.3)  # Dropout layer with 50% dropout rate after fc2
        self.fc3 = nn.Linear(64, 32)  # Another hidden layer with 32 units
        self.fc4 = nn.Linear(32, output_size)  # Output layer with 3 units for 3 classes


    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1).to(dtype=torch.float32)

        # Pass through the layers with ReLU activation for hidden layers and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after activation of fc1
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after activation of fc2
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Raw logits

        return F.softmax(x)  # Return raw logits

