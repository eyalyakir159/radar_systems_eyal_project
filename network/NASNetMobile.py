import torch
import torch.nn as nn
import pretrainedmodels
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights


class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMobileNetV2, self).__init__()

        # Load the pre-trained MobileNetV2 model using the weights parameter
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Modify the input layer to accept 4 channels
        self.model.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Modify the last fully connected layer to match the number of classes
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x=x.to(dtype=torch.float32)
        return self.model(x)



