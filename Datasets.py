from Util import imshow
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import torch
import tqdm
import torchattacks as attacks
import numpy as np
import itertools

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class BaseDataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.ToTensor()

        rootdir = "./traindata" if train else "./testdata"
        self.trainset = torchvision.datasets.CIFAR10(root=rootdir, train=train,
                                                download=False, transform=self.transform)

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        return self.trainset[idx]


# Inputs: images with pixels changed. Labels: tensor indicating which pixels changed.
class PixelRandomizationDataset(Dataset):
    def __init__(self, train, orig_labels=False):
        self.base_dataset = BaseDataset(train)

        self.new_set = []

        # randomly select 15% of pixels to change (32x32=1024 --> 1024*.15 ~= 154)
        # with replacement so might be slightly less
        print("generating data...")
        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            points_changed = [0 for i in range(0,1024)] # list to store changed pixels
            this_pic = torch.clone(self.base_dataset[idx][0]) # new distorted image
            label = self.base_dataset[idx][1]
            for _ in range(0, 154):
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

            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = (this_pic, points_changed)
            self.new_set.append(tup)
        print("done generating data")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]


class PatchDataset(Dataset):
    """
    For this dataset, create 16 patches of 8x8 pixels.
    Randomly select 3 patches to be omitted from the input image.
    Of these 3 patches, mildly distort 2. TODO: change to positional embedding instead
    The target is a one hot vector indicating which is the non distorted patch missing from the input image.
    TODO: should be x = 13 original patches, y = 3 original patches, target = one hot vec indicating which of the 3 patches belongs in the position of the original image indicated by the one-hot vector
    """
    def __init__(self, train, orig_labels=False):
        self.base_dataset = BaseDataset(train)

        self.new_set = []

        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            this_pic = torch.clone(self.base_dataset[idx][0])
            label = self.base_dataset[idx][1]

            # pick which indeces will be y and which x
            indeces = random.sample([n for n in range(0, 16)], k = 3)
            target = [1, 0, 0]
            random.shuffle(target)
            x = []
            y = []
            for r in range(0,4):
                for c in range(0,4):
                    s = this_pic[0][r*8:r*8+8,c*8:c*8+8]
                    b = this_pic[1][r*8:r*8+8,c*8:c*8+8]
                    g = this_pic[2][r*8:r*8+8,c*8:c*8+8]
                    patch = torch.stack((s, b, g), dim=0)
                    if 4*r+c in indeces:
                        y.append(patch)
                    else:
                        x.append(patch)

            for i in range(0, len(target)):
                value = target[i]
                patch = y[i]

                # TODO: get rid of this and add in positional embedding of chosen patch instead
                # distort 2 of the patches slightly
                if not value:
                    for _ in range(0, 10): # distort 10 of 64 pixels
                        # randomly select x and y value
                        m = random.randint(0, 7)
                        n = random.randint(0, 7)

                        # three values to change
                        r = random.uniform(-.1, .1)
                        b = random.uniform(-.1, .1)
                        g = random.uniform(-.1, .1)

                        # change r
                        patch[0][m][n] = patch[0][m][n] + r
                        # change b
                        patch[1][m][n] = patch[1][m][n] + b
                        # change g
                        patch[2][m][n] = patch[2][m][n] + g

                y[i] = patch

            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = ([x, y], torch.Tensor(target))
            self.new_set.append(tup)
        print("done generating data")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]


class JigsawDataset(Dataset):
    """
    For this dataset, create 9 patches of 10x10 pixels.
    Randomly select a permutation from a set of possible permutations
    Reorder the patches into this permutation
    The target is a one hot vector indicating the index of the permutation from the set.
    """
    def __init__(self, train, orig_labels=False):
        self.base_dataset = BaseDataset(train)
        perms = list(set(itertools.permutations([0,1,2,3,4,5,6,7,8])))
        random.shuffle(perms)
        self.permutations = perms[:64] # these are the possible permutations
        self.new_set = []

        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            this_pic = torch.clone(self.base_dataset[idx][0])
            label = self.base_dataset[idx][1]
            index = random.randint(0, len(self.permutations)-1)
            perm = self.permutations[index]
            target = [0] * len(self.permutations)
            target[index] = 1

            # generate patches
            patches = []
            for r in range(0,3):
                for c in range(0,3):
                    s = this_pic[0][r*10+r:r*10+10+r,c*10+c:c*10+10+c]
                    b = this_pic[1][r*10+r:r*10+10+r,c*10+c:c*10+10+c]
                    g = this_pic[2][r*10+r:r*10+10+r,c*10+c:c*10+10+c]
                    patch = torch.stack((s, b, g), dim=0)
                    patches.append(patch)

            input = []
            for i in perm:
                input.append(patches[i])


            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = (torch.stack(input), torch.Tensor(target))
            self.new_set.append(tup)
        print("done generating data")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]


class AttackDataset(Dataset):
    def __init__(self, base_dataset, network, attack):
        self.base_dataset = base_dataset

        self.new_set = []

        if attack == "PGD":
            attack_fn = attacks.PGD(network, eps=8/255, alpha=2/225, steps=100, random_start=True)
        else:
            raise ValueError("Not a valid attack type")

        print("generating adversarial examples...")

        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            this_pic = torch.clone(self.base_dataset[idx][0]) # new distorted image
            label = self.base_dataset[idx][1]

            adv_image = attack_fn(torch.unsqueeze(this_pic, dim=0), torch.tensor([label]))

            self.new_set.append((adv_image.squeeze(), label))

        # print(classes[self.base_dataset[0][1]])
        # imshow(self.base_dataset[0][0])

        # print(classes[network(torch.unsqueeze(self.new_set[0][0], dim=0))])
        # imshow(self.new_set[0][0])

        print("done generating adversarial examples")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]


# Add more datasets here as they are made
def get_dataloader(dataset_type, train, batch_size, network, attack):
    if dataset_type == "Base":
        dataset = BaseDataset(train)
    elif dataset_type == "PixelRandomization":
        dataset = PixelRandomizationDataset(train)
    elif dataset_type == "PatchFill":
        dataset = PatchDataset(train)
    elif dataset_type == "Jigsaw":
        dataset = JigsawDataset(train)
    else:
        raise ValueError("Not a valid dataset type")

    if not train and attack is not None:
        dataset = AttackDataset(dataset, network, attack)

    return torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=True, num_workers=2)


#get_dataloader("Jigsaw", True, 1, None, False)