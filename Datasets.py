from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import torch
import tqdm
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, train):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        print("generating data...")

        # randomly select 15% of pixels to change (32x32=1024 --> 1024*.15 ~= 154)
        # with replacement so might be slightly less
        print("generating data...")
        for image in tqdm.tqdm(range(0, len(self.base_dataset)), total=len(self.base_dataset)):
            points_changed = [0 for i in range(0,1024)] # list to store changed pixels
            this_pic = torch.clone(self.base_dataset[image][0]) # new distorted image
            label = self.base_dataset[image][1]
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


# Add more datasets here as they are made
def get_dataloader(dataset_type, train, batch_size):
    if dataset_type == "Base":
        dataset = BaseDataset(train)
    elif dataset_type == "PixelRandomization":
        dataset = PixelRandomizationDataset(train)
    else:
        raise ValueError("Not a valid dataset type")

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)