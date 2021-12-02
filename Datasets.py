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

            # pick which indeces will be y and which x
            indeces = random.sample([n for n in range(0, 9)], k = 3)
            target = [1, 0, 0]
            random.shuffle(target)
            x = []
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
                    if 3*r+c in indeces:
                        if target[indeces.index(3*r+c)]:
                            y_pos = torch.Tensor(position)
                        y.append(patch)
                    else:
                        x.append((patch, torch.Tensor(position)))

            if orig_labels:
                tup = (this_pic, label)
            else:
                tup = ([x, y, y_pos], torch.Tensor(target))
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

# transforms for contrastive learning
    #     transforms = torch.nn.Sequential(
    #     transforms.RandomResizedCrop(size=size),
    #     transforms.RandomHorizontalFlip(),
    #     # transforms.RandomApply([color_jitter], p=0.8),
    #     transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
    #     transforms.RandomGrayscale(p=0.2),
    #     # GaussianBlur(kernel_size=int(0.1 * size)),
    # )


class AttackDataset(Dataset):
    def __init__(self, base_dataset, network, attack):
        self.base_dataset = base_dataset

        self.new_set = []

        if attack == "FGSM":
            attack_fn = attacks.FGSM(network, eps=8/255)
        elif attack == "PGD50":
            attack_fn = attacks.PGD(network, eps=8/255, alpha=2/225, steps=50, random_start=True)
        elif attack == "PGD200":
            attack_fn = attacks.PGD(network, eps=8/255, alpha=2/225, steps=200, random_start=True)
        elif attack == "BIM":
            attack_fn = attacks.BIM(network, eps=8/255, alpha=2/255, steps=50)
        elif attack == "AutoAttack":
            attack_fn = attacks.AutoAttack(network, eps=8/255, n_classes=10, version='standard')
        elif attack == "CW":
            attack_fn = attacks.CW(network, c=1, lr=0.01, steps=50, kappa=0)
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