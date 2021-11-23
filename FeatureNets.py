from Util import Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch
from Util import Normalize

"""
Modules for pretraining tasks

In addition to a forward method, these modules should also implement a "get_features"
method that makes an intermediate representation available that can be incorporated
into another network.

As such, "get_feature_size" should also be defined to return the number of features per image.
"""

class DistortedPixelNet(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1024)

        self.feature_size = 84

    def get_features(self, x):
        x = self.norm(x)
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x):
        x = self.get_features(x)
        return self.fc3(x)

    def get_feature_size(self):
        return self.feature_size