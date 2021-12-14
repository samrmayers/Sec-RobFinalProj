from Util import imshow, imshow_gray, gray
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import torch
import tqdm
import torchattacks as attacks
import numpy as np
import itertools
import matplotlib.image as mpimg

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
    For this dataset, create 9 patches of 10x10 pixels.
    Randomly select 3 patches to be omitted from the input image.
    The target is a one hot vector indicating which is the indicated patch missing from the input image.
    should be x = 6 original patches, y = 3 original patches, target = one hot vec indicating which of the 3 patches belongs in the position of the original image indicated by the one-hot vector
    """
    def __init__(self, train, orig_labels=False):
        self.base_dataset = BaseDataset(train)

        self.new_set = []

        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            this_pic = torch.clone(self.base_dataset[idx][0])
            label = self.base_dataset[idx][1]

            # pick which indices will be y and which x
            indices = random.sample([n for n in range(0, 9)], k = 3)
            target = [1, 0, 0]
            random.shuffle(target)
            x = []
            x_pos = []
            y = []
            y_pos = []
            for r in range(0, 3):
                for c in range(0, 3):
                    s = this_pic[0][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    b = this_pic[1][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    g = this_pic[2][r * 10 + r:r * 10 + 10 + r, c * 10 + c:c * 10 + 10 + c]
                    patch = torch.stack((s, b, g), dim=0)
                    position = [0]*9
                    position[3*r+c] = 1
                    if 3*r+c in indices:
                        if target[indices.index(3*r+c)]:
                            y_pos = torch.Tensor(position) # positional embedding of patch indicated in target vector
                        y.append(patch)
                    else:
                        x.append(patch)
                        x_pos.append(torch.Tensor(position))

            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = ([torch.stack(x, dim=0), torch.stack(x_pos, dim=0), torch.stack(y, dim=0), y_pos], torch.Tensor(target)) # x are the 6 patches + positions, y is 3 other patches, y_pos is 1 x 9 is correct position
            self.new_set.append(tup)
        print("done generating data")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]

class ColorDataset(Dataset):
    """
        For this dataset, create a black and white image and have original be the target.
        """

    def __init__(self, train, orig_labels=False):
        self.base_dataset = BaseDataset(train)

        self.new_set = []

        for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            this_pic = torch.clone(self.base_dataset[idx][0])
            label = self.base_dataset[idx][1]
            blackandwhite = gray(this_pic)
            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = (blackandwhite, this_pic)
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
        self.permutations = list(set(itertools.permutations([0,1,2,3])))
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
            pos = []
            for r in range(0,2):
                for c in range(0,2):
                    s = this_pic[0][r*15+r:r*15+15+r,c*15+c:c*15+15+c]
                    b = this_pic[1][r*15+r:r*15+15+r,c*15+c:c*15+15+c]
                    g = this_pic[2][r*15+r:r*15+15+r,c*15+c:c*15+15+c]
                    patch = torch.stack((s, b, g), dim=0)
                    patches.append(patch)
                    position = [0] * 4
                    position[2 * r + c] = 1
                    pos.append(torch.Tensor(position))

            input = []
            for i in perm:
                input.append(patches[i])

            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = ((torch.stack(input), torch.stack(pos)), torch.Tensor(target))
            self.new_set.append(tup)
        print("done generating data")

    def __len__(self):
        return len(self.new_set)

    def __getitem__(self, idx):
        return self.new_set[idx]

# transforms for contrastive learning
    #     transforms = torch.nn.Sequential(
    #     transforms.RandomResizedCrop(size=size),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomApply([color_jitter], p=0.8),
    #     transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
    #     transforms.RandomGrayscale(p=0.2),
    #     # GaussianBlur(kernel_size=int(0.1 * size)),
    # )

class ContrastiveDataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32,32), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

        rootdir = "./traindata" if train else "./testdata"
        self.trainset = torchvision.datasets.CIFAR10(root=rootdir, train=train,
                                                download=False, transform=self.transform)

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        return self.trainset[idx]

class AdvContrastiveDataset(Dataset):
    def __init__(self, train, network, batch_size):
        rootdir = "./traindata" if train else "./testdata"
        self.base_dataset = torchvision.datasets.CIFAR10(root=rootdir, train=train,
                                                download=False, transform=self.transform)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu = torch.device("cpu")
        self.network = network.to(device)

        sample_im, _ = self.base_dataset[0]

        self.new_ims = torch.empty((0, *(sample_im.shape)))
        self.new_labels = torch.empty((0))

        attack_fn = attacks.FGSM(self.network, eps=2/255)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32,32), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

        self.dataloader = torch.utils.data.DataLoader(self.base_dataset, batch_size=batch_size, num_workers=2)

        # Replace half the images with adversarial versions
        for images, labels in self.dataloader:
            indices = torch.randperm(batch_size)[:batch_size/2]
            adv_images = attack_fn(images[indices].to(device), labels[indices].to(device)).to(cpu)

            self.new_ims = torch.cat((self.new_ims, adv_images), 0)
            self.new_labels = torch.cat((self.new_labels, labels[indices]), 0)

            self.new_ims = torch.cat((self.new_ims, images[~indices]), 0)
            self.new_labels = torch.cat((self.new_labels, labels[~indices]), 0)

            print(len(self.new_ims))

    # Needs to return examples, corresponding adversarial examples (as a pair), and a mask

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        return self.trainset[idx]

class SelfContrastiveDataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(32,32), scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])

        rootdir = "./traindata" if train else "./testdata"
        self.base_dataset = torchvision.datasets.CIFAR10(root=rootdir, train=train, download=False)

    def __get_pair__(self, idx):
        this_image_raw, _ = self.base_dataset[idx]

        t1 = self.transform(this_image_raw)
        t2 = self.transform(this_image_raw)

        return (t1, t2), torch.tensor(0)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        return self.__get_pair__(idx)


class AttackDataset(Dataset):
    def __init__(self, base_dataset, network, attack, batch_size):
        self.base_dataset = base_dataset
        sample_im, _ = self.base_dataset[0]

        self.new_ims = torch.empty((0, *(sample_im.shape)))
        self.new_labels = torch.empty((0))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cpu = torch.device("cpu")

        self.network = network.to(device)

        if attack == "FGSM":
            attack_fn = attacks.FGSM(self.network, eps=2/255)
        elif attack == "PGD10":
            attack_fn = attacks.PGD(self.network, eps=2/255, alpha=2/225, steps=10, random_start=True)
        elif attack == "PGD50":
            attack_fn = attacks.PGD(self.network, eps=2/255, alpha=1/225, steps=50, random_start=True)
        elif attack == "BIM":
            attack_fn = attacks.BIM(self.network, eps=2/255, alpha=1/255, steps=50)
        elif attack == "AutoAttack":
            attack_fn = attacks.AutoAttack(self.network, eps=8/255, n_classes=10, version='standard')
        else:
            raise ValueError("Not a valid attack type")

        print("generating adversarial examples...")

        self.dataloader = torch.utils.data.DataLoader(self.base_dataset, batch_size=batch_size, num_workers=2)

        for images, labels in self.dataloader:
        # for idx in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
        #     this_pic = torch.clone(self.base_dataset[idx][0]) # new distorted image
        #     label = self.base_dataset[idx][1]


            adv_images = attack_fn(images.to(device), labels.to(device)).to(cpu)

            self.new_ims = torch.cat((self.new_ims, adv_images), 0)
            self.new_labels = torch.cat((self.new_labels, labels), 0)

            print(len(self.new_ims))

        # print(classes[self.base_dataset[0][1]])
        # imshow(self.base_dataset[0][0])

        # print(classes[network(torch.unsqueeze(self.new_set[0][0], dim=0))])
        # imshow(self.new_set[0][0])

        print("done generating adversarial examples")

    def __len__(self):
        return len(self.new_ims)

    def __getitem__(self, idx):
        return (self.new_ims[idx], self.new_labels[idx])


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
    elif dataset_type == "Colorizer":
        dataset = ColorDataset(train)
    elif dataset_type == "ColorizerNew":
        dataset = ColorDataset(train)
    elif dataset_type == "Contrastive":
        dataset = ContrastiveDataset(train)
    elif dataset_type == "SelfContrastive":
        dataset = SelfContrastiveDataset(train)
    elif dataset_type == "AdvContrastive":
        dataset = AdvContrastiveDataset(train, network)
    else:
        raise ValueError("Not a valid dataset type")

    if not train and attack is not None:
        dataset = AttackDataset(dataset, network, attack, batch_size)

    return torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=True, num_workers=2)


#get_dataloader("Color", True, 1, None, False)
