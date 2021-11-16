from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import torch
import tqdm
import matplotlib.pyplot as plt
import numpy as np

# this method displays an image - same method as in Trainer.py
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# ImageDataset class generates:
# 1. "Normal" image data (original) --> Inputs: original unchanged images. Labels: original class labels.
# 2. Pretrain data --> Inputs: images with pixels changed. Labels: tensor indicating which pixels changed.
# 3. Perturbed images --> Inputs: images with pixels changed. Labels: orginal class labels.
class ImageDataset(Dataset):
    def __init__(self, rootdir, train=True, pretrain=False, perturbed=False):
        self.pretrain = pretrain
        self.perturbed = perturbed
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root=rootdir, train=train,
                                                download=False, transform=self.transform)

        self.new_set = [] # only necessary for pretrain=True or perturbed=True

        # randomly select 15% of pixels to change (32x32=1024 --> 1024*.15 ~= 154)
        # with replacement so might be slightly less
        if pretrain:
            print("generating data...")
            for image in tqdm.tqdm(range(0, len(self.trainset)), total=len(self.trainset)):
                points_changed = [0 for i in range(0,1024)] # list to store changed pixels
                this_pic = torch.clone(self.trainset[image][0]) # new distorted image
                for pixel_number in range(0, 154):
                    # randomly select x and y value
                    x = random.randint(0,31)
                    y = random.randint(0,31)
                    point = x*32 + y
                    points_changed[point] = 1

                    # three values to change
                    r = random.uniform(-1,1)
                    b = random.uniform(-1,1)
                    g = random.uniform(-1,1)

                    # change r
                    this_pic[0][x][y] = r
                    # change b
                    this_pic[1][x][y] = b
                    # change g
                    this_pic[2][x][y] = g
                points_changed = torch.Tensor(points_changed)
                tup = (this_pic, points_changed) # new example
                self.new_set.append(tup)
            print("done generating pretrain data")

        if self.perturbed and not train:
            print("generating data...")
            for image in tqdm.tqdm(range(0, len(self.trainset)), total=len(self.trainset)):
                # randomly decide if perturbed or not
                p = random.choice([0,1,1])
                print(p)
                this_pic = torch.clone(self.trainset[image][0])
                label = self.trainset[image][1]
                if p:
                    for pixel_number in range(0, 154):
                        # randomly select x and y value
                        x = random.randint(0,31)
                        y = random.randint(0,31)

                        # three values to change
                        r = random.uniform(-1,1)
                        b = random.uniform(-1,1)
                        g = random.uniform(-1,1)

                        # change r
                        this_pic[0][x][y] = r
                        # change b
                        this_pic[1][x][y] = b
                        # change g
                        this_pic[2][x][y] = g
                tup = (this_pic, label) # new example
                self.new_set.append(tup)
            print("done generating data")

    def __len__(self):
        if self.pretrain or self.perturbed:
            return len(self.new_set)
        return len(self.trainset)

    def __getitem__(self, idx):
        if self.pretrain or self.perturbed:
            return self.new_set[idx]
        return self.trainset[idx]