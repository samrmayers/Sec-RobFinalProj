from Util import Normalize
import torch.nn as nn
import torch.nn.functional as F
import torch
from Util import Normalize, Identity
import torchvision
from ResNets import *

"""
Modules for pretraining tasks

In addition to a forward method, these modules should also implement a "get_features"
method that makes an intermediate representation available that can be incorporated
into another network.

As such, "get_feature_size" should also be defined to return the number of features per image.
"""

class ResNet18(nn.Module):
    def __init__(self, feature_networks, num_features):
        super().__init__()

        if len(feature_networks) > 0 or num_features > 0:
            raise ValueError("ResNet18 doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.rescale = torchvision.transforms.Resize((224,224))

        self.resnet = resnet18(num_classes=1000, pretrained=True, progress=True)

        self.resnet.fc = Identity()

        self.feature_size = 512
        self.fc = nn.Linear(512, 10)

    def get_features(self, x):
        x = self.norm(x)
        x = self.rescale(x)
        return self.resnet(x)

    def forward(self, x):
        x = self.get_features(x)
        return self.fc(x)

    def get_feature_size(self):
        return self.feature_size

class ResNet50(nn.Module):
    def __init__(self, feature_networks, num_features):
        super().__init__()

        if len(feature_networks) > 0 or num_features > 0:
            raise ValueError("ResNet18 doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.rescale = torchvision.transforms.Resize((224,224))

        self.resnet = resnet50(num_classes=1000, pretrained=True, progress=True)

        self.resnet.fc = Identity()

        self.feature_size = 4096
        self.fc = nn.Linear(4096, 10)

    def get_features(self, x):
        x = self.norm(x)
        x = self.rescale(x)
        return self.resnet(x)

    def forward(self, x):
        x = self.get_features(x)
        return self.fc(x)

    def get_feature_size(self):
        return self.feature_size

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
        self.conv2_bn = nn.BatchNorm2d(16)
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

class SelfieNet(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        # patch processing network
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)

        # decoder
        self.fc3 = nn.Linear(84, 3)

        # patch pos embedding
        self.pos = nn.Linear(9,84)

        self.feature_size = 84

    def patch_processing(self, x):
        x = self.norm(x)
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def get_features(self, x):
        allx = []
        for patch in x:
            with_pos = torch.add(self.patch_processing(patch[0]), self.pos(patch[1]))
            allx.append(with_pos)
        u = torch.stack(allx)
        u = u.sum(dim=0)
        return u

    def forward(self, w):
        x = w[0]
        y = w[1]
        allx = []
        for patch in x:
            with_pos = torch.add(self.patch_processing(patch[0]), self.pos(patch[1])) #TODO: check that adding pos is ok? 99% sure this is right but might as well check
            allx.append(with_pos)
        u = torch.stack(allx)
        u = u.sum(dim=0) #TODO: change to attention pooling network
        ally = []
        for patch in y:
            with_pos = torch.add(self.patch_processing(patch[0]), self.pos(patch[1]))
            ally.append(torch.diagonal(torch.matmul(with_pos,torch.transpose(u, 0, 1))))
        result = F.softmax(torch.stack(ally), dim=1)
        return torch.transpose(result, 0, 1)

    def get_feature_size(self):
        return self.feature_size

class JigsawNet(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        # patch processing network
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 64)

        self.feature_size = 84

    def patch_processing(self, x):
        x = self.norm(x)
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def get_features(self, x):
        return self.patch_processing(x)

    def forward(self, x):
        allx = []
        for patch in x:
            allx.append(self.patch_processing(patch))
        u = torch.stack(allx)
        result = F.softmax(self.fc3(u), dim=2).sum(dim=1) #TODO: check this pooling
        return result

    def get_feature_size(self):
        return self.feature_size
