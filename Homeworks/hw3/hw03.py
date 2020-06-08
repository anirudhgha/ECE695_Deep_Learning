import torch.nn as nn
import torchvision.transforms as tvt
import torchvision
import torch
import torch.nn.functional as F
import numpy as np



def hw03_code():
    cifar_root = 'C:\\Users\\alasg\\Downloads\\cifar-10-python.tar\\cifar-10-python'
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


    class TemplateNet(nn.Module):
        def __init__(self, conv_layers, padding):
            super(TemplateNet, self).__init__()
            self.padding = padding
            self.conv_layers = conv_layers
            if self.conv_layers == 2 and self.padding == 0:
                self.fcsize = 6*6*128
            elif self.conv_layers == 2 and self.padding == 1:
                self.fcsize = 7*7*128
            elif self.conv_layers == 1 and self.padding == 0:
              self.fcsize = 15*15*128
            elif self.conv_layers == 1 and self.padding == 1:
              self.fcsize = 16*16*128
            

            self.conv1 = nn.Conv2d(3, 128, 3, padding=self.padding)  ## (A)
            if self.conv_layers == 2:
                self.conv2 = nn.Conv2d(128, 128, 3)  ## (B)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(self.fcsize, 1000)  ## (C)
            self.fc2 = nn.Linear(1000, 10)
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            if self.conv_layers == 2:
                x =  self.pool(F.relu(self.conv2(x))) ## (D)
            x = x.view(-1, self.fcsize)  ## (E)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def run_code_for_training(net, task):
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

            # outputs is a 4 x 10 tensor
            if task == 4:
                for jj in range(4):
                    val, ind = outputs[jj].max(0)
                    confusion[ind,labels[jj]] += 1
                values, indices = outputs[0].max(0)
            

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 11999 == 0 and i != 0:
                f.write("[epoch:%d, batch:%5d] loss: %.3f\n" % (epoch + 1, i + 1, running_loss / float(2000)))
                print("[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(2000)))
                if(task == 4):
                    print(confusion)
                    f.write(str(confusion))
            if i % 2000 == 1999:
                running_loss = 0.0

    def run_code_for_testing(net, task):
        epochs = 1
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        confusion = torch.zeros(10, 10)
        for epoch in range(epochs):
            running_loss = 0.0
        for i, data in enumerate(test_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            # outputs is a 4 x 10 tensor
            if task == 4:
                for jj in range(4):
                    val, ind = outputs[jj].max(0)
                    confusion[ind,labels[jj]] += 1
                values, indices = outputs[0].max(0)
            
            
        print(confusion)
        f.write(str(confusion))
       


    net_task1 = TemplateNet(conv_layers=1, padding=0)
    net_task2 = TemplateNet(conv_layers=2, padding=0)
    net_task3 = TemplateNet(conv_layers=2, padding=1)

    run_code_for_training(net_task1, 1)
    run_code_for_training(net_task2, 2)
    run_code_for_training(net_task3, 3)
    run_code_for_testing(net_task3, 4)
    
    f.close()

if __name__ == '__main__':
    hw03_code()