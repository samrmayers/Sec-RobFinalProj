import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# freezes a module so its weights aren't updated during the rest of the model training
def freeze_module(module):
    for layer in module.children():
        for param in layer.parameters():
            param.requires_grad = False


# displays an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# turns an image black and white
def gray(img):
    transform = torchvision.transforms.Grayscale(num_output_channels=3)
    return transform(img)

# displays black and white image
def imshow_gray(img):
    img = gray(img)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Normalization layer for networks, from https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
class Normalize(nn.Module):
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, x):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (x - mean) / std

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
