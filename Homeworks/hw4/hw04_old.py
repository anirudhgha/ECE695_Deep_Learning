import torch.nn as nn
import torchvision.transforms as tvt
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
# from .resnet import *
from resnet import ResNet


def hw04_code():
    print('start')
    cifar_root = 'C:\\Users\\alasg\\Downloads\\cifar-10-python.tar\\cifar-10-python\\'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    f = open("output.txt", 'w')

    # read in cifar-10
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)
    test_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform)

    # Now create the data loaders:
    batch_size, num_epochs = 4, 1

    train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    # This is what we pass to the ResNet class to build us a resnet. All the other layers don't require user input.
    class BasicBlock1(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock1, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.expansion = 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    class BasicBlock2(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock2, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.expansion = 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            # adding a relu to function output before concatenating with identity
            out = self.relu(out)

            out += identity
            out = self.relu(out)

            return out

    def run_code_for_training(net):
        epochs = 1
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        confusion = torch.zeros(10, 10)
        for epoch in range(epochs):
            running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 999 == 0 and i != 0:
                f.write("[epoch:%d, batch:%5d] loss: %.3f\n" % (epoch + 1, i + 1, running_loss / float(2000)))
                print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
            if i % 999 == 0:
                running_loss = 0.0

    def run_code_for_testing(net):
        net = net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        confusion = torch.zeros(10, 10)
        running_loss = 0
        for i, data in enumerate(test_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss

            # outputs is a 4 x 10 tensor
            for jj in range(4):
                val, ind = outputs[jj].max(0)
                confusion[ind, labels[jj]] += 1

        print(confusion)
        f.write(str(confusion))

        acc = 0
        for i in range(len(confusion)):
            acc += confusion[i, i]
        acc = (acc / torch.sum(confusion)) * 100
        f.write('Classification accuracy: {} '.format(acc))

    # define some resnet architecture
    net_skip1 = ResNet(BasicBlock1, layers=[2, 2, 2, 2], num_classes=10)
    net_skip2 = ResNet(BasicBlock2, layers=[2, 2, 2, 2], num_classes=10)

    run_code_for_training(net_skip1)
    run_code_for_testing(net_skip1)

    run_code_for_training(net_skip2)
    run_code_for_testing(net_skip2)

    f.close()


if __name__ == '__main__':
    hw04_code()