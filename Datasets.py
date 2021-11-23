from Util import imshow
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import torch
import tqdm
import torchattacks as attacks
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class BaseDataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.ToTensor()

        rootdir = "./traindata" if train else "./testdata"
        self.trainset = torchvision.datasets.CIFAR10(root=rootdir, train=train,
                                                download=True, transform=self.transform)

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
    else:
        raise ValueError("Not a valid dataset type")

    if not train and attack is not None:
        dataset = AttackDataset(dataset, network, attack)

    return torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=True, num_workers=2)