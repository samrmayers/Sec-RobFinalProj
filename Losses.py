import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_metric_learning import losses

# from https://github.com/pytorch/pytorch/issues/1249
# https://towardsdatascience.com/choosing-and-customizing-loss-functions-for-image-processing-a0e4bf665b0a

# from above ^ said this loss was good for tensor overlap
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        smooth = 1.

        iflat = output.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))

# Based on https://www.kaggle.com/debarshichanda/pytorch-supervised-contrastive-learning/notebook
# From this paper: https://arxiv.org/abs/2004.11362
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, output, target):
        # take only first 128 features
        output = output[:, 0:128]

        # Normalize feature vectors
        output_normalized = F.normalize(output, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                output_normalized, torch.transpose(output_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=self.temperature)(logits, torch.squeeze(target))

class ContrastiveJointLoss(nn.Module):
    def __init__(self, weight = 1):
        super().__init__()
        self.contrastive_loss = SupervisedContrastiveLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, output, target):
        last_features = output[:, 128:138]

        loss_c = self.contrastive_loss(output, target)
        loss_l = self.cross_entropy_loss(last_features, target)
        return loss_c + self.weight*loss_l

# Inspired by https://spell.ml/blog/simple-contrastive-learning-representation-using-the-X7QycRIAACQAqLu-
class SelfContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, output, target):
        # target is a dummy here

        z_i = F.normalize(output[0][:, 0:128], dim=1)
        z_j = F.normalize(output[1][:, 0:128], dim=1)

        batch_size = len(z_i)
        negatives_mask =(~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


def get_loss(task):
    if task == "BasicClassification":
        return nn.CrossEntropyLoss()
    elif task == "PixelRandomization":
        return DiceLoss()
    elif task == "PatchFill":
        return nn.CrossEntropyLoss()
    elif task == "Jigsaw":
        return nn.CrossEntropyLoss()
    elif task == "Colorizer":
        return nn.MSELoss()
    elif task == "ColorizerNew":
        return nn.MSELoss()
    elif task == "ResNet18":
        return nn.CrossEntropyLoss()
    elif task == "ResNet50":
        return nn.CrossEntropyLoss()
    elif task == "AggNet":
        return nn.CrossEntropyLoss()
    elif task == "Contrastive":
        return SupervisedContrastiveLoss()
    elif task == "SelfContrastive":
        return SelfContrastiveLoss()
    elif task == "JointContrastive":
        return ContrastiveJointLoss(weight=20)
    else:
        raise ValueError("No loss specified for this task")