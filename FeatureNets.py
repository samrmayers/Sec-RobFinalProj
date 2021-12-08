from Util import Normalize, gray
import torch.nn as nn
import torch.nn.functional as F
import torch
from Util import Normalize, Identity
import torchvision
from ResNets import *
import numpy as np

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

        self.feature_size = 2048
        self.fc = nn.Linear(2048, 10)

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

        self.rescale = torchvision.transforms.Resize((224, 224))
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

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
        x = self.norm(x)
        x = self.rescale(x)
        x = self.net(x) # output size will be batch x <something> x 512
        return x

    def pos_processing(self, y_pos):
        return F.relu(self.pos(y_pos))

    def x_processing(self, x, x_pos): # x is tuples for each patch, patch[0] is patch, patch[1] is pos

        # run x through self.net, batch x patches x 3 x 10 x 10 -> batch x patches x 512 (may need to flatten patches x batch into a pseudo batch and then reverse)
        allx = []
        for i in range(x.shape[1]):
            patch = x[:, i, ...]
            temp = self.patch_processing(patch)
            allx.append(temp)

        # run x_pos through self.pos_processing, batch x patches x 9 -> batch x patches x 512
        allx_pos = []
        for i in range(x_pos.shape[1]):
            patch = x_pos[:, i, ...]
            temp = self.pos_processing(patch)
            allx_pos.append(temp)

        # add x, x_pos
        allx = torch.stack(allx, dim=1)
        allx = allx.squeeze()
        allx_pos = torch.stack(allx_pos, dim=1)
        u = torch.add(allx, allx_pos)

        # torch.sum in the dim = 1 (would eventually be the attention step), batch x 512
        u = u.sum(dim=1)  # TODO: change to attention pooling network
        return u

    def get_features(self, x):

        # break image into 9 patches
        newx = []
        x_pos = []
        for pic in x:
            this_pic = torch.clone(pic)
            for r in range(0, 3):
                for c in range(0, 3):
                    s = this_pic[0][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    b = this_pic[1][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    g = this_pic[2][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    patch = torch.stack((s, b, g), dim=0)
                    position = [0]*9
                    position[3*r+c] = 1
                    newx.append(patch)
                    x_pos.append(torch.Tensor(position))

        # put all 9 patches w/ positions through x_processing
        new = torch.stack(newx, dim = 1) # should be batches x patches x 3 x 10 x 10
        newpos = torch.stack(x_pos, dim = 1)
        return self.x_processing(new, newpos)

    def forward(self, w):
        # x dims are batch x 3 x 32 x 32

        # ---------- processing steps -----------------

        x = w[0]  # batchx6x3x10x10
        x_pos = w[1]
        y = w[2]
        y_pos = w[3]

        # run x, y through self.net, batch x patches x 3 x 10 x 10 -> batch x patches x 512 (may need to flatten patches x batch into a pseudo batch and then reverse)
        u = self.x_processing(x, x_pos)

        # (ends here for feature net / get_features)

        # ---------- task specific steps -----------------

        ally = []
        for i in range(y.shape[1]):
            patch = y[:, i, ...]
            ally.append(self.patch_processing(patch))
        h = torch.stack(ally, dim=1)
        h = h.squeeze()

        # run x_pos, y_pos through self.pos_processing, batch x patches x 9 -> batch x patches x 512
        y_pos = self.pos_processing(y_pos)

        # then add y_pos, batch x 512
        v = torch.add(u, y_pos)

        # unsqueeze the above in dim 1, replicate 3 times -> batch x 3 x 512
        v.unsqueeze(dim = 1)
        vs = torch.stack([v, v, v], dim=1)

        # dot in dim 2 with y, softmax, -> batch x 3 (cross entropy with target)
        result = np.empty(shape=(vs.shape[0], vs.shape[1]))
        for i,batch in enumerate(vs):
            for j,v in enumerate(batch):
                temp = torch.dot(v,h[i][j])
                result[i][j] = temp

        result = torch.tensor(result)
        result = F.softmax(result, dim=1)
        return result

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

class JigsawNetNew(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        self.rescale = torchvision.transforms.Resize((224, 224))
        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # resnet (first 3 layers encoder, last decoder)
        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = Identity()

        # task
        self.fc = nn.Linear(512, 24)

        self.feature_size = 512

    def patch_processing(self, x):
        # resize to batch ( x patches ) x 224 x 224

        x = self.norm(x)
        x = self.rescale(x)
        x = self.net(x) # output size will be batch x <something> x 512

        return x

    def x_processing(self, x):
        # x is tuples for each patch, patch[0] is patch, patch[1] is pos
        shape = x.shape[0:2]

        # run x through self.net, batch x patches x 3 x 10 x 10 -> batch x patches x 512 (may need to flatten patches x batch into a pseudo batch and then reverse)       x = x.flatten((0,1))
        x = torch.flatten(x, 0,1)
        x = self.patch_processing(x).squeeze().unflatten(0, shape).sum(dim=1)
        # torch.sum in the dim = 1 (would eventually be the attention step), batch x 512
        return x

    def get_features(self, x):

        # break image into 9 patches
        newx = []
        for pic in x:
            this_pic = torch.clone(pic)
            for r in range(0, 3):
                for c in range(0, 3):
                    s = this_pic[0][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    b = this_pic[1][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    g = this_pic[2][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    patch = torch.stack((s, b, g), dim=0)
                    newx.append(patch)

        # put all 9 patches w/ positions through x_processing
        newx = torch.stack(newx, dim=1) # should be batches x patches x 3 x 10 x 10
        return self.x_processing(newx)

    def forward(self, x):
        # x dims are batch x 3 x 32 x 32

        # ---------- processing steps -----------------

        x = self.x_processing(x)

        # (ends here for feature net / get_features)

        # ---------- task specific steps -----------------

        x = F.softmax(self.fc(x), dim=1)
        return x

    def get_feature_size(self):
        return self.feature_size

class ColorizerNet(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.rescale = torchvision.transforms.Resize((224, 224))

        self.resnet = resnet18(num_classes=1000, pretrained=True, progress=True)
        self.resnet.fc = Identity()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(8, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, 1),
            nn.Upsample((32, 32)),
            nn.Tanh(),
        )

        self.feature_size = 512

        self.gray = torchvision.transforms.Grayscale(num_output_channels=3)

        # input is batch x 3 x 32 x 32

        # ------- preprocessing -------------

        # greyscale, batch x 32 x 32

        # ----------- processing ------------

        # pass through network, batch x feature_size

        # --------- task specific -------------

        # reconstruction x upsampling layers, batch x feature_size -> batch x 3 x 32 x 32

    def get_features(self, x):
        x = self.norm(x)
        x = self.gray(x)
        x = self.rescale(x)
        return self.resnet(x)

    def forward(self, x):
        x = self.get_features(x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = torch.reshape(x,(x.shape[0], 8, 8, 8))
        x = self.model(x)
        return x

    def get_feature_size(self):
        return self.feature_size

class ColorizerNetNew(nn.Module):
    def __init__(self, feature_nets, num_features):
        super().__init__()

        if len(feature_nets) > 0 or num_features > 0:
            raise ValueError("PixelDistortion doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Beta-version/beta_version.ipynb
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv7 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv9 = nn.Conv2d(256, 128, 3, 1, 1)
        self.upsample = nn.Upsample((2,2))
        self.conv10 = nn.Conv2d(128, 64, 3, 1, 1)
        # upsample again
        self.conv11 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv12 = nn.Conv2d(32, 3, 3, 1, 1)
        # tanh
        #upsample
        self.upsamplefinal = nn.Upsample((32, 32))

        self.avg = nn.AvgPool2d(4)
        self.feature_size = 512

    def get_features(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.avg(x).squeeze()
        return x

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.upsample(x)
        x = F.relu(self.conv10(x))
        x = self.upsample(x)
        x = F.relu(self.conv11(x))
        x = torch.tanh(self.conv12(x))
        x = self.upsamplefinal(x)
        return x

    def get_feature_size(self):
        return self.feature_size

class ContrastiveNet(nn.Module):
    """
    This net is used for contrastive learning-eqsue tasks

    get_features: batch x 3 x 32 x 32 -> batch x 512

    forward: batch x 3 x 32 x 32 -> batch x (128 + 10)(two more layers,
        split to give both a representation for contrastive and cross entropy)
    """
    def __init__(self, feature_networks, num_features):
        super().__init__()

        if len(feature_networks) > 0 or num_features > 0:
            raise ValueError("ContrastiveNet doesn't accept feature networks")

        self.norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.rescale = torchvision.transforms.Resize((224,224))

        self.resnet = resnet18(num_classes=1000, pretrained=True, progress=True)

        self.resnet.fc = Identity()

        self.feature_size = 512
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, 10)

    def get_features(self, x):
        x = self.norm(x)
        x = self.rescale(x)
        return self.resnet(x)

    def forward(self, x):

        x = self.get_features(x)
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        y = self.fc3(x)

        x = torch.cat((z,y), dim=1)

        return x

    def get_feature_size(self):
        return self.feature_size
