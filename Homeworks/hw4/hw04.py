import torch.nn as nn
import torchvision.transforms as tvt
# from import torchvision.models.resnet import ResNet
import torchvision
import torch
import torch.nn.functional as F
import numpy as np
# from resnet import ResNet
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def hw04_code():
    # cifar_root = 'C:\\Users\\alasg\\Downloads\\cifar-10-python.tar\\cifar-10-python\\'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    f = open("output.txt", 'w')

    # parameters
    image_size = 128
    batch_size, num_epochs = 10, 1

    # cifar 10
    # transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # imagenet

    transform = tvt.Compose(
        [tvt.Resize(size=image_size), tvt.CenterCrop(size=image_size), tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)
    # test_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform)

    # build a custom dataset using downloaded imagenet images
    dataroot_laptop_train = r'C:\Users\alasg\Documents\Course work\ECE695_DL\Homeworks\hw4\imagenet_images\Train_dataset'
    dataroot_laptop_test = r'C:\Users\alasg\Documents\Course work\ECE695_DL\Homeworks\hw4\imagenet_images\Test_dataset'
    data_root_train = r'./drive/My Drive/Colab Notebooks/ece695_Deep_Learning/hw4/imagenet_images/Train_dataset'
    data_root_test = r'./drive/My Drive/Colab Notebooks/ece695_Deep_Learning/hw4/imagenet_images/Test_dataset'

    train_data_loc = ImageFolder(root=dataroot_laptop_train, transform=transform)
    test_data_loc = ImageFolder(root=dataroot_laptop_test, transform=transform)

    # Now create the data loaders:
    train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # THE FOLLOWING CODE FOR CONV3X3, CONV1X1, AND THE BASICBLOCK FUNCTIONS ARE BASED ON PYTORCH'S PUBLIC RESNET CODE,
    # WITH MODIFICATIONS MADE BOTH TO SIMPLIFY AND EXTEND THE DEFAULT CODE.

    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes, out_planes, stride=1):
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
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            # adding a relu to function output before concatenating with identity
            # out = self.relu(out)

            out += identity
            out = self.relu(out)

            return out

    # Variant 3 conduct batch normalization on both the summed function and identity rather than on just the function.
    class BasicBlock3(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock3, self).__init__()
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

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            # bn shifted from inside F(x) to F(x)+x
            out = self.bn2(out)
            out = self.relu(out)

            return out

    #  # Variant 4: Adding a padding layer that extends the boundaries of the input, and a maxpool layer to the end of
    #     # the skipblock, before the final relu function. The additional padding should keep the size of the input the same
    #     # over multiple layers

    class BasicBlock4(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock4, self).__init__()
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
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
            self.padding = nn.ReflectionPad2d(2)

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

            # additional padding and maxpool layers
            out = self.padding(out)
            out = self.maxpool(out)
            out = self.relu(out)

            return out

    # THE FOLLOWING CODE FOR RESNET, MAKE_LAYER, FORWARD_IMPL AND FORWARD ARE BASED ON PYTORCH'S PUBLIC RESNET CODE,
    # WITH MODIFICATIONS MADE TO SIMPLIFY PYTORCH'S CODE.
    class ResNet(nn.Module):

        def __init__(self, block, layers, num_classes=5):
            super(ResNet, self).__init__()
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
            self.inplanes = 64
            self.dilation = 1
            replace_stride_with_dilation = [False, False, False]
            self.groups = 1
            self.base_width = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)


        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation

            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def _forward_impl(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x

        def forward(self, x):
            return self._forward_impl(x)

    def run_code_for_training(net, batch_size, num_epochs):
        epochs = num_epochs
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        confusion = torch.zeros(10, 10)
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0
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

                # if i % 200 == 199 and i != 0:
                #     f.write("[epoch:%d, batch:%5d] loss: %.3f\n" % (epoch + 1, i + 1, running_loss / float(200)))
                #     print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(200)))
                # if i % 200 == 199:
                #     running_loss = 0.0
                num_batches += 1
            f.write("Epoch %d: %.3f\n" % (epoch + 1, running_loss / float(num_batches)))
            print("Epoch %d: %.3f" % (epoch + 1, running_loss / float(num_batches)))

    def run_code_for_testing(net, batch_size):
        net = net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        # confusion = torch.zeros(10, 10)
        running_loss = 0
        num_batches=0
        correct = 0
        total = 0
        for i, data in enumerate(test_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            num_batches += 1

            # compare predicted vs actual to tally correct predictions
            for jj in range(len(outputs)):
                val, ind = outputs[jj].max(0)
                if ind == labels[jj]:
                    correct += 1
                total += 1

        print('Classification accuracy: ', (correct/total)*100)
        f.write('Classification accuracy: {} '.format((correct/total)*100))

    # define some resnet architecture
    net_skip1 = ResNet(BasicBlock1, layers=[1, 1, 1, 1], num_classes=5)
    net_skip2 = ResNet(BasicBlock2, layers=[2, 2, 2, 2], num_classes=5)
    net_skip3 = ResNet(BasicBlock3, layers=[2, 2, 2, 2], num_classes=5)
    net_skip4 = ResNet(BasicBlock4, layers=[2, 2, 2, 2], num_classes=5)


    # Variation 1: Default resnet architecture using BasicBlock from resnet18 and resnet34.
    # net_skip1 = torchvision.models.resnet18(pretrained=True)
    run_code_for_training(net_skip1, batch_size, num_epochs)
    run_code_for_testing(net_skip1, batch_size)

    # Variation 2: This variation on resnet has an extra conv/bn/relu block between the 
    # first and second convolution blocks of the default resnet18 architecture. Making it more similar to the bottleneck
    # structure, but without downsampling and upsampling between convolutional layers.
    # run_code_for_training(net_skip2, batch_size, num_epochs)
    # run_code_for_testing(net_skip2, batch_size)

    # Variation 3: This variation on resnet has the batch normalization layer shifted to after the identity and
    # function are summed together.
    # run_code_for_training(net_skip3, batch_size, num_epochs)
    # run_code_for_testing(net_skip3, batch_size)

    # Variant 4: Adding a padding layer that extends the boundaries of the input, and a maxpool layer to the end of
    # the skipblock, before the final relu function. The additional padding should keep the size of the input the same
    # over multiple layers
    # run_code_for_training(net_skip4, batch_size, num_epochs)
    # run_code_for_testing(net_skip4, batch_size)

    f.close()


if __name__ == '__main__':
    hw04_code()