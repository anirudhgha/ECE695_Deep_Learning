import torch
import torch.nn as nn
import torchvision.transforms as tvt
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def hw04_code():
    # cifar_root = 'C:\\Users\\alasg\\Downloads\\cifar-10-python.tar\\cifar-10-python\\'
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    f = open("output.txt", 'w')

    # cifar 10
    # transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # imagenet
    image_size = 128
    transform = tvt.Compose(
        [tvt.Resize(size=image_size), tvt.CenterCrop(size=image_size), tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=True, transform=transform)
    # test_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=True, transform=transform)

    # build a custom dataset using downloaded imagenet images
    dataroot_laptop_train = r'C:\Users\alasg\Documents\Course work\ECE695_DL\Homeworks\hw4\imagenet_images\Train_dataset'
    dataroot_laptop_test = r'C:\Users\alasg\Documents\Course work\ECE695_DL\Homeworks\hw4\imagenet_images\Test_dataset'
    dataroot_train = r'./drive/My Drive/Colab Notebooks/ece695_Deep_Learning/hw4/imagenet_images/Train_dataset'
    dataroot_test = r'./drive/My Drive/Colab Notebooks/ece695_Deep_Learning/hw4/imagenet_images/Test_dataset'

    train_data_loc = ImageFolder(root=dataroot_laptop_train, transform=transform)
    test_data_loc = ImageFolder(root=dataroot_laptop_test, transform=transform)

    # Now create the data loaders:
    batch_size, num_epochs = 10, 20

    train_data_loader = torch.utils.data.DataLoader(train_data_loc, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(test_data_loc, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # THE FOLLOWING CODE FOR THE CLASS SKIPBLOCK IS BASED ON PROF. AVINASH KAK'S DLSTUDIO.CUSTOMDATALOADING.SKIPBLOCK
    # CLASS. THE CODE WAS MODIFIED TO BUILD VARIANTS OF THE ORIGINAL SKIPBLOCK CLASS.
    # Variant 1, default skipblock (VARIANT 1 IS PROF. AVINASH KAK'S CODE AS IS TO ENSURE SOME WORKING STARTING POINT)
    class SkipBlock1(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipBlock1, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.convo(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity
            return out

    # Variant 2, implements an additional convolution/bn/relu layer after the first round in each skipblock.
    # this affects the F(x) and leaves the x the same, where the output of the skipblock is defined as F(x) + x.
    class SkipBlock2(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipBlock2, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.convo(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)

            if self.in_ch == self.out_ch:
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)

                # Variant 2: adding a second iteration of convolution/bn/relu as long as inchannel == outchannel
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)


            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity
            return out

    # Variation 3: This variation on resnet has the batch normalization layer shifted to after the identity and
    # function are summed together. This should normalize both the identity and the function output, which may affect
    # classification accuracy.
    class SkipBlock3(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipBlock3, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.convo(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)

            if self.in_ch == self.out_ch:
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity

            # bn shifted from inside F(x) to F(x)+x
            out = self.bn2(out)
            return out

    # Variant 4: Adding a padding layer that extends the boundaries of the input, and a maxpool layer to the end of
    # the skipblock, before the final relu function. The additional padding should keep the size of the input the same
    # over multiple layers
    class SkipBlock4(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(SkipBlock4, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
            self.padding = nn.ReflectionPad2d(1)

        def forward(self, x):
            identity = x
            out = self.convo(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity

            out = self.padding(out)
            out = self.maxpool(out)
            return out



    # THE FOLLOWING BMENET CLASS IS BASED ON PROF. AVINASH KAK'S DLSTUDIO.CUSTOMDATALOADING.BMENET CLASS. THE CODE
    # WAS MODIFIED TO BUILD A MORE SHALLOW NETWORK TO GO ALONG WITH THE SMALLER, CUSTOM IMAGENET DATASET, THOUGH
    # STILL USING SKIP CONNECTIONS
    class BMEnet(nn.Module):
        def __init__(self, depth=32):
            super(BMEnet, self).__init__()
            self.pool_count = 3
            self.depth = depth // 2
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip64 = SkipBlock4(64, 64)
            self.skip64ds = SkipBlock4(64, 64, downsample=True)
            self.skip64to128 = SkipBlock4(64, 128)
            self.skip128 = SkipBlock4(128, 128)
            self.skip128ds = SkipBlock4(128, 128, downsample=True)
            self.fc1 = nn.Linear(128 * (image_size // 2 ** self.pool_count) ** 2, 1000)
            self.fc2 = nn.Linear(1000, 5)

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            x = self.skip64(x)
            x = self.skip64ds(x)
            x = self.skip64(x)
            x = self.skip64to128(x)
            x = self.skip128(x)
            x = self.skip128ds(x)
            x = self.skip128(x)
            x = x.view(-1, 128 * (image_size // 2 ** self.pool_count) ** 2)
            x = torch.nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

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
                #     f.write("[epoch:%d, batch:%5d] loss: %.3f\n" % (epoch + 1, i + 1, running_loss / float(num_batches)))
                #     print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(num_batches)))
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


    #----------------------------------------------- BMEnet Variants ---------------------------------------------------
    # Variation 1: Default resnet architecture using BasicBlock from resnet18 and resnet34.

    # Variation 2: This variation on resnet has an extra conv/bn/relu block between the
    # first and second convolution blocks of the default resnet18 architecture. Making it more similar to the bottleneck
    # structure, but without downsampling and upsampling between convolutional layers.

    # Variation 3: This variation on resnet has the batch normalization layer shifted to after the identity and
    # function are summed together. This should normalize both the identity and the function output, which may affect
    # classification accuracy.

    # Variant 4: Adding a padding layer that extends the boundaries of the input, and a maxpool layer to the end of
    # the skipblock, before the final relu function. The additional padding should keep the size of the input the same
    # over multiple layers. This may help further 'normalize' the data (with an additional maxpool) which may help the
    # network generalize better to testing data.

    # final run uses variant 4, which appends a padding layer and maxpool layer to the end of skipblock to
    # better generalize to testing data. Testing between the variants actually showed varient 4 to have slightly better
    # classification accuracy by 1-2%, and so it's been used as the final output. 
    net_skip = BMEnet()
    run_code_for_training(net_skip, batch_size, num_epochs)
    run_code_for_testing(net_skip, batch_size)

    f.close()


if __name__ == '__main__':
    hw04_code()