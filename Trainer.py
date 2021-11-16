import torch
import torchvision
import torchvision.transforms as transforms
import Models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ImageDataset import ImageDataset
from DiceLoss import DiceLoss
import argparse

# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
PATH = './cifar_net.pth'
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# this method displays an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# main training loop. Can change number of epochs --> maybe make this a parameter later
# learning rate scheduler not helping right now --> can come back to this later
def train(model, loss_fn, optimizer, trainloader, scheduler=False):
    if scheduler == True:
        sched = optim.lr_scheduler.StepLR(optimizer,25)

    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        print("EPOCH: ", epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler==True:
                sched.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model

# set up for training. Initializes trainset, trainloader, calls main training loop, saves model
def train_and_save(path=PATH,pretrain=False):
    if pretrain:
        criterion = DiceLoss() # can come back to this later, can be changed depending on what output we use for pretraining
    else:
        criterion = nn.CrossEntropyLoss() # used for training classification
    if pretrain:
        model = Models.Classifier() # pretraining
    else:
        model = Models.Net() # main classification
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    batch_size = 16 # can change batchsize --> maybe add in as a parameter

    trainset = ImageDataset(rootdir='./traindata', train=True, pretrain=pretrain)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)


    model = train(model, criterion, optimizer, trainloader, scheduler=False)
    torch.save(model.state_dict(), path)

# method to load a model and test it. Can test on "perturbed" data used for pretraining task.
def load_and_test(path=PATH, perturbed = False):
    net = Models.Net() # if testing pretraining need to change this to net = Models.Classifier() for correct dimensions
    net.load_state_dict(torch.load(path))

    batch_size = 2

    testset = ImageDataset(rootdir='./testdata', train=False, pretrain=False, perturbed=perturbed)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

# main
def main(args):
    train = (args.train == 'True')
    pretrain = (args.pretrain == 'True')
    perturbed = (args.perturbed == 'True')
    if train:
        train_and_save(path=args.path,pretrain=pretrain)
    else:
        load_and_test(path=args.path, perturbed=perturbed)

# arguments needed to run
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',dest='path',required=True,
                        help='The pathway to 1. Save a new model or 2. Load a model.')
    parser.add_argument('--train',dest='train', required=True,
                        choices=["True", "False"],
                        help='Whether to train a model or test a model.')
    parser.add_argument('--pretrain',dest='pretrain', required=False,
                        choices=["True", "False"],
                        help='Whether to do pretraining task or not. For training only.')
    parser.add_argument('--perturbed',dest='perturbed', required=False,
                        choices=["True", "False"],
                        help='Whether to pass perturbed data into model.')
    args = parser.parse_args()
    main(args)


