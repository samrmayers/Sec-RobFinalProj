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

    def pos_processing(self, y_pos):
        return F.relu(self.pos(y_pos))

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
        y_pos = w[2]
        allx = []
        for patch in x:
            with_pos = torch.add(self.patch_processing(patch[0]), self.pos(patch[1])) #TODO: check that adding pos is ok? 99% sure this is right but might as well check
            allx.append(with_pos)
        u = torch.stack(allx)
        u = u.sum(dim=0)  # TODO: change to attention pooling network
        y_pos = self.pos_processing(y_pos) # change size of positional embedding to 84 so its the same as u
        v = torch.cat([u, y_pos]) # concat to have v (patches + pos embedding of missing patch)
        ally = []
        for patch in y:
            temp = self.patch_processing(patch)
            ally.append(torch.diagonal(torch.matmul(temp,torch.transpose(v, 0, 1))))
        result = F.softmax(torch.stack(ally), dim=1)
        return torch.transpose(result, 0, 1)

    def get_feature_size(self):
        return self.feature_size

class SelfieNetNew(nn.Module):
    """
    All the patches are processed by the same patch processing network P .
    On the encoder side, the output vectors produced by P are routed into the attention pooling network
    to summarize these representations into a single vector u.
    On the decoder side, P creates output vectors h1, h2, h3.
    The decoder then queries the encoder by adding to the output vector u the location embedding of a patch,
    selected at random among the patches in the decoder (e.g., location4) to create a vector v.
    The vector v is then used in a dot product to compute the similarity between v and each h.
    Having seen the dot products between v and h’s, the decoder has to decide which patch is most relevant
    to fill in the chosen location (at location4).
    The cross entropy loss is applied for this classification task,
    whereas the encoder and decoder are trained jointly with gradients back-propagated from this loss.
    """
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        # resnet (first 3 layers encoder, last decoder)
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        self.layers = list(self.pretrained._modules.keys())
        self.finetune = self.pretrained._modules.pop(self.layers[-1])
        self.net = nn.Sequential(self.pretrained._modules) # should be first 3 layers
        self.pretrained = None

        # patch pos embedding
        self.pos = nn.Linear(9,512)

        self.feature_size = 512

    def patch_processing(self, x):

        # resize to batch ( x patches ) x 224 x 224
        x = self.net(x) # output size will be batch x <something> x 512
        return x

    def pos_processing(self, y_pos):
        return F.relu(self.pos(y_pos))

    def get_features(self, x):
        # TODO: fix this

        # x dims are batch x 3 x 32 x 32

        # ---------- preprocessing steps -----------------
        # break x into patches batch x 6 x 3 x 10 x 10

        # get corresponding x_pos as batch x 6 x 9

        # remaining patches y as batch x 3 x 3 x 10 x 10

        # get corresponding y_pos as batch x 9

        # ---------- processing steps -----------------

        # run x, y through self.net, batch x patches x 3 x 10 x 10 -> batch x patches x 512 (may need to flatten patches x batch into a pseudo batch and then reverse)

        # run x_pos, y_pos through self.pos_processing, batch x patches x 9 -> batch x patches x 512

        # add x, x_pos

        # torch.sum in the dim = 1 (would eventually be the attention step), batch x 512

        # (ends here for feature net / get_features)

        # ---------- task specific steps -----------------

        # then add y_pos, batch x 512

        # unsqueeze the above in dim 1, replicate 3 times -> batch x 3 x 512

        # dot in dim 2 with y_pos, softmax, -> batch x 3 (cross entropy with target)

        return x

    # TODO: fix this to work
    def forward(self, w):
        x = w[0] # batchx6x3x10x10
        y = w[1]
        y_pos = w[2]
        allx = []
        # for each patch, do patch processing + add the positional embedding for that patch
        for patch in x:
            with_pos = torch.add(self.patch_processing(patch[0]), self.pos_processing(patch[1])) #TODO: check that adding pos is ok? 99% sure this is right but might as well check
            allx.append(with_pos)
        u = torch.stack(allx)
        # pooling to get vector u
        u = u.sum(dim=0)  # TODO: change to attention pooling network
        y_pos = self.pos_processing(y_pos)
        # add positional embedding of the "chosen" patch to u to get vector v
        v = torch.add(u, y_pos) # add to have v (patches + pos embedding of missing patch)
        ally = []
        # for each "distractor" patch and the chosen patch, do patch processing.
        # for each of these vectors, dot with v to get similarity.
        for patch in y:
            temp = self.patch_processing(patch)
            print("TEMP SIZE", temp.shape)
            print("V SIZE", torch.transpose(v, 1, 0).shape)
            ally.append(torch.diagonal(torch.dot(temp,torch.transpose(v, 1, 0))))
        # softmax each result to get which patch is correct
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

class ColorizerNet(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Alpha-version/alpha_version_notebook.ipynb

        # input is batch x 3 x 32 x 32

        # ------- preprocessing -------------

        # greyscale, batch x 32 x 32

        # ----------- processing ------------

        # pass through network, batch x feature_size

        # --------- task specific -------------

        # reconstruction x upsampling layers, batch x feature_size -> batch x 3 x 32 x 32
