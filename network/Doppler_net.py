import torch
import torch.nn as nn
import torch.optim as optim


class DopplerNet(nn.Module):
    def __init__(self,in_channels , out_channels):
        super(DopplerNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1,
                               padding=1)  # Adjust in_channels based on your input

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Adjust the input_features based on your flattened feature map size
        self.fc1 = nn.Linear(in_features=32 * 10 * 61 , out_features=128)  # Example: 32 filters, input size: 10x61
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=out_channels)  # Adjust out_features based on your number of classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x=x.to(dtype=torch.float32)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x


