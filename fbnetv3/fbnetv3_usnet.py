from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from ptflops import get_model_complexity_info

from fbnetv3_abs import AbsUnderEvalNet
from fbnetv3_utils import Candidate
from mbconv import BlockArgs, MBConvBlock, calculate_output_image_size


def select_layer(layer_name, layer_params, image_size, in_channels):
    if layer_name == 'Conv':
        return (Conv2dRELU(in_channels=in_channels, out_channels=layer_params['c'],
                           kernel_size=layer_params['k'], stride=layer_params['s']),
                new_image_size(image_size, layer_params['k'], layer_params['s']),
                layer_params['c'])
    elif layer_name == 'MBConv':
        block_args = BlockArgs(in_channels, layer_params['c'], layer_params['k'],
                               layer_params['s'], layer_params['e'], 0 if layer_params['se'] is False else 1)
        return (MBConvBlock(block_args=block_args, image_size=image_size),
                calculate_output_image_size(image_size, layer_params['s']),
                layer_params['c'])
    elif layer_name == 'Pool':
        if layer_params['class'] == 'Max':
            pool = nn.MaxPool2d
        elif layer_params['class'] == 'Avg':
            pool = nn.AvgPool2d
        return (pool(layer_params['k']),
                new_image_size(image_size, layer_params['k'], layer_params['k']),
                in_channels)

def new_image_size(old_size, k_s, stride):
    new_size = list()
    if isinstance(k_s, int):
        k_s = [k_s for i in range(len(old_size))]
    if stride is None:
        stride = [1 for i in range(len(old_size))]
    elif isinstance(stride, int):
        stride = [stride for i in range(len(old_size))]
    for i in range(len(old_size)):
        new_size.append((old_size[i] - k_s[i]) // stride[i] + 1)
    return new_size


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class Conv2dRELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dRELU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return F.relu(self.conv(x))


class SearchNet(nn.Module):
    def __init__(self, args: dict, image_size=(32, 32), in_channels=3):
        super(SearchNet, self).__init__()
        stages = OrderedDict()
        stage_names = ['stage1', 'stage2', 'stage3', 'stage4']
        for stage_name in stage_names:
            layer_name, layer_params = list(args[stage_name].items())[0]
            stages[stage_name], image_size, in_channels = select_layer(layer_name, layer_params, image_size, in_channels)
            stages['{}_relu'.format(stage_name)] = nn.ReLU()
        self.stages = nn.Sequential(stages)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(torch.prod(torch.tensor(image_size)).item() * in_channels, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.stages(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


class UnderEvalNet(AbsUnderEvalNet):
    def __init__(self, candidate: Candidate):
        self.candidate = candidate
        self.net = SearchNet(candidate.o_data)

    def set_statistics_label(self):
        self.candidate.statistics_label = [num / 10000 for num in get_model_complexity_info(self.net, (3, 32, 32), as_strings=False, print_per_layer_stat=False)]

    def set_acc_label(self, total_epoch: int, record_acc_each_epoch=False):
        print('[eval candidate] {}'.format(self.candidate.o_data), flush=True)

        if record_acc_each_epoch is True:
            self.candidate.acc_each_epoch = [1 for _ in range(total_epoch)]

        args = self.candidate.o_data['recipe']
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        criterion = nn.CrossEntropyLoss()
        if args['optim'] == 'SGD':
            optimizer = torch.optim.SGD(self.net.parameters(), lr=args['lr'] * 0.0004, momentum=0.9, weight_decay=args['wd'] * 0.000001)
        elif args['optim'] == 'RMSProp':
            optimizer = torch.optim.RMSprop(self.net.parameters(), lr=args['lr'] * 0.001, momentum=0.9, weight_decay=args['wd'] * 0.000001)

        for epoch in range(total_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[eval candidate] [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000), flush=True)
                    running_loss = 0.0

        print('Finished Training', flush=True)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total), flush=True)
        self.candidate.acc_label = [correct / total]


if __name__ == "__main__":
    args = {'stage1': {'Conv': {'k': 5, 'c': 6, 's': 1}}, 'stage2': {'Pool': {'k': 2, 'class': 'Max'}}, 'stage3': {'Conv': {'k': 5, 'c': 16, 's': 1}}, 'stage4': {'Pool': {'k': 2, 'class': 'Max'}}, 'recipe': {'lr': 2.5, 'optim': 'SGD', 'wd': 0}}
    pass
