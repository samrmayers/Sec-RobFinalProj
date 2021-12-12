import torch
import torch.optim as optim
import argparse

from Datasets import get_dataloader
from Losses import get_loss
from ClassifierNets import make_network

# from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
PATH = './cifar_net.pth'
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(network, trainloader, main_task, main_path, epochs, scheduler_period):
    if main_task == "Colorizer":
        optimizer = optim.RMSprop(network.parameters(), lr=0.01, momentum=0.9)
    else:
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

    loss_fn = get_loss(main_task)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    #print(next(network.parameters()).is_cuda)

    if scheduler_period > 0:
        sched = optim.lr_scheduler.StepLR(optimizer,scheduler_period)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print("EPOCH: ", epoch)
        for i, data in enumerate(trainloader, 0):
            if main_task == "Jigsaw":
                patches = data[0][0].to(device)
                pos = data[0][1].to(device)
                labels = data[1].to(device)
                inputs = (patches, pos)
            elif main_task == "PatchFill":
                xpatches = data[0][0].to(device)
                xpos = data[0][1].to(device)
                ypatches = data[0][2].to(device)
                ypos = data[0][3].to(device)
                labels = data[1].to(device)
                inputs = (xpatches, xpos, ypatches, ypos)
            else:
                inputs, labels = data[0].to(device), data[1].to(device)
            #if i == 0:
              #print(inputs.is_cuda)
              #print(labels.is_cuda)
            #inputs, labels = data[0], data[1] #for testing on cpu
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler_period > 0:
                sched.step()

            running_loss += loss.item()
            if i % 200 == 199:  # print every 90 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        torch.save(network.state_dict(), main_path)

    print('Finished Training')


# This function assumes a classification network
def test(net, testloader):
    correct = 0
    total = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Network accuracy on test set: %d %%' % (100 * correct / total))


def main(args):
    should_train = (args.train == 'True')

    network = make_network(args.main_task, args.main_path, args.feature_nets, should_train)
    dataloader = get_dataloader(args.dataloader, should_train, args.batch_size, network, args.attack)

    if should_train:
        train(network, dataloader, args.main_task, args.main_path, args.epochs, args.scheduler_period)
    else:
        test(network, dataloader)


"""
Arguments:
    train               -- whether to train the specified model or test it.
    main_task           -- which network architecture to use for the main network
    main_path           -- location for main network weights
    feature_nets        -- types and locations of additional networks, specified
                            in order separated by spaces
    epochs              -- number of epochs for training
    scheduler_period    -- LR scheduler step size for training, not used if 0
    batch_size          -- batch size for training or testing
    dataloader          -- type of dataloader
    attack              -- type of attack to use in testing
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',dest='train', required=True,
                        choices=["True", "False"],
                        help='Whether to train a model or test a model.')
    parser.add_argument('--main_task',dest='main_task',required=True,
                        help='The name of the main task architecture.')
    parser.add_argument('--main_path',dest='main_path',required=True,
                        help='The pathway to save or load weights for the main task architecture.')
    parser.add_argument('--feature_nets',dest='feature_nets',required=False, nargs="*",
                        default=[],
                        help='The paired names and paths of any pretrained feature networks to use with the main task.')
    parser.add_argument('--epochs',dest='epochs', required=False, type=int,
                        default=20,
                        help='The number of epochs to train for.')
    parser.add_argument('--scheduler_period',dest='scheduler_period', required=False, type=int,
                        default=0,
                        help='The learning rate scheduler period, or 0 if not used')
    parser.add_argument('--batch_size',dest='batch_size', required=False, type=int,
                        default=4,
                        help='The batch size for training or testing.')
    parser.add_argument('--dataloader',dest='dataloader',required=True,
                        help='The name of the dataloader to use in training or testing.')
    parser.add_argument('--attack',dest='attack',required=False,
                        default=None,
                        help='The attack to use in testing')
    args = parser.parse_args()
    main(args)
