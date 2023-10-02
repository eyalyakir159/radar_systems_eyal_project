import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self,in_channels , out_channels):
        super(CustomCNN, self).__init__()


        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 1 * 7, 128)  # Adjust the input size based on the output size of your conv layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)  # 4 output classes: car, person, drone

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input shape: [batch_size, 1, 10, 61] assuming grayscale image
        x=x.to(dtype=torch.float32)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Downsample by a factor of 2

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Downsample by a factor of 2

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Downsample by a factor of 2

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
