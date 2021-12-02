import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from Util import freeze_module, Normalize
from FeatureNets import *
from ResNets import *

"""
Modules for main classification tasks.

These modules should be defined to take in an arbitrary list of FeatureNets (along
with the total number of output features for all of them) which can be used to generate
additional features from the input image before the final classifier. These FeatureNets
should have their weights frozen.

The "build_network" function at the bottom should be updated to handle new architectures
for classifiers and feature networks as they are added.
"""

# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class BasicNet(nn.Module):
    def __init__(self, feature_networks, num_features):
        super().__init__()
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

        self.feature_networks = feature_networks
        for net in self.feature_networks:
            freeze_module(net)

        self.num_features = num_features
        self.fc3 = nn.Linear(84 + self.num_features, 10)

    def forward(self, x):
        x = self.norm(x)
        x_orig = x.clone()
        x = self.conv1_bn(self.pool(F.relu(self.conv1(x))))
        x = self.conv2_bn(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        for net in self.feature_networks:
            x = torch.cat((x, net.get_features(x_orig)), 1)

        x = self.fc3(x)
        return x

class AggNet(nn.Module):
    def __init__(self, feature_networks, num_features):
        super().__init__()

        if len(feature_networks) == 0 or num_features == 0:
            raise ValueError("JustFeaturesNet needs some feature networks")

        self.feature_networks = feature_networks
        for net in self.feature_networks:
            freeze_module(net)

        self.num_features = num_features
        self.fc1 = nn.Linear(self.num_features, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = None
        for net in self.feature_networks:
            if out is None:
                out = net.get_features(x)
            else:
                out = torch.cat((out, net.get_features(x)), 1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Add more networks here as they are added
def get_network(task):
    if task == "BasicClassification":
        return BasicNet
    elif task == "PixelRandomization":
        return DistortedPixelNet
    elif task == "PatchFill":
        return SelfieNet
    elif task == "Jigsaw":
        return JigsawNet
    elif task == "ResNet18":
        return ResNet18
    elif task == "ResNet50":
        return ResNet50
    elif task == "AggNet":
        return AggNet
    else:
        raise ValueError(f"Invalid network type {task}")

def make_network(main_task, main_path, feature_nets, training):
    if len(feature_nets) % 2 != 0:
        raise ValueError("Feature Network specified without a path")
    task_path_pairs = zip(feature_nets[::2], feature_nets[1::2])

    nets = []
    num_features = 0
    for task, path in task_path_pairs:
        net = get_network(task)([],0)

        net.load_state_dict(torch.load(path))
        nets.append(net)
        num_features += net.get_feature_size()

    main_net = get_network(main_task)(nets, num_features)
    if not training:
        main_net.load_state_dict(torch.load(main_path))
    return main_net