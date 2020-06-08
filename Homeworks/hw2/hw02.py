import torchvision.transforms as tvt
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

import DLStudio


def hw02_code():
    f = open("output.txt", "w")

    cifar_root = 'C:\\Users\\alasg\\Downloads\\cifar-10-python.tar\\cifar-10-python'

    # DLS=DLStudio(dataroot = './cifar-10-batches-py/')

    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=True, download=False, transform=transform)
    test_data_loc = torchvision.datasets.CIFAR10(root=cifar_root, train=False, download=False, transform=transform)

    # data loaders need only to load cat and dog data
    data_catdog = []
    targets_catdog = []
    data_catdog_test = []
    targets_catdog_test = []

    for i in range(len(train_data_loc)):
        if train_data_loc.targets[i] == 3 or train_data_loc.targets[i] == 5:
            data_catdog.append(train_data_loc.data[i])
            targets_catdog.append(train_data_loc.targets[i])
    for i in range(len(test_data_loc)):
        if test_data_loc.targets[i] == 3 or test_data_loc.targets[i] == 5:
            data_catdog_test.append(test_data_loc.data[i])
            targets_catdog_test.append(test_data_loc.targets[i])

    train_data_loc.data = np.asarray(data_catdog)
    train_data_loc.targets = np.asarray(targets_catdog)
    test_data_loc.data = np.asarray(data_catdog_test)
    test_data_loc.targets = np.asarray(targets_catdog_test)

    train_data_loc.classes = ["cat", "dog"]
    test_data_loc.classes = ["cat", "dog"]

    # Now create the data loaders:
    batch_size, num_epochs = 5, 100
    trainloader = torch.utils.data.DataLoader(train_data_loc, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data_loc, batch_size=batch_size, shuffle=False, num_workers=2)

    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    N, D_in, H1, H2, D_out = 8, 3 * 32 * 32, 1000, 256, 2

    # Randomly initialize weights
    w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
    w2 = torch.randn(H1, H2, device=device, dtype=dtype)
    w3 = torch.randn(H2, D_out, device=device, dtype=dtype)

    learning_rate = 1e-9
    loss_store = [0 for i in range(num_epochs)]

    for t in range(num_epochs):
        printed = False
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y = [[0 for i in range(2)] for x in range(5)]
            for i in range(batch_size):
                if labels[i] == 5:
                    y[i][:] = [0, 1]  # dog
                else:
                    y[i][:] = [1, 0]  # cat

            y = torch.FloatTensor(y)
            y = y.to(device)

            # 5 samples down, 5 images accross
            x = inputs.view(batch_size, D_in)
            h1 = x.mm(w1)
            h1_relu = h1.clamp(min=0)
            h2 = h1_relu.mm(w2)
            h2_relu = h2.clamp(min=0)
            y_pred = h2_relu.mm(w3)

            # Compute and print loss
            loss = (y_pred - y).pow(2).sum().item()
            if not printed:
                f.write("Epoch {}: {}\n".format(t, loss))
                print(t, loss)
                loss_store[t] = loss
            printed = True

            y_error = y_pred - y
            grad_w3 = h2_relu.t().mm(2 * y_error)  # <<<<<< Gradient of Loss w.r.t w3
            h2_error = 2.0 * y_error.mm(w3.t())  # backpropagated error to the h2 hidden layer
            h2_error[h2_error < 0] = 0  # We set those elements of the backpropagated error
            grad_w2 = h1_relu.t().mm(2 * h2_error)  # <<<<<< Gradient of Loss w.r.t w2
            h1_error = 2.0 * h2_error.mm(w2.t())  # backpropagated error to the h1 hidden layer
            h1_error[h1_error < 0] = 0  # We set those elements of the backpropagated error
            grad_w1 = x.t().mm(2 * h1_error)  # <<<<<< Gradient of Loss w.r.t w2

            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
            w3 -= learning_rate * grad_w3

    loss_sum = 0
    total_num = 0
    correct = 0
    # test the network now that training is complete
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        y = [[0 for i in range(2)] for x in range(batch_size)]
        for ii in range(batch_size):
            if labels[ii] == 5:
                y[ii][:] = [0, 1]  # dog
            else:
                y[ii][:] = [1, 0]  # cat

        y = torch.FloatTensor(y)
        y = y.to(device)

        x = inputs.view(batch_size, D_in)
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)

        for ii in range(batch_size):
            if labels[ii] == 3 and y_pred[ii, 0] > y_pred[ii, 1]:
                correct += 1
            elif labels[ii] == 5 and y_pred[ii, 1] > y_pred[ii, 0]:
                correct += 1

        # Compute and print loss
        loss_sum = (y_pred - y).pow(2).sum().item()
        total_num += batch_size

    print('Testing accuracy: ', correct / total_num)

    f.write("\nTest Accuracy: {}%".format((correct / total_num) * 100))
    f.close()

    # plt.plot(np.arange(5), loss_store, label='Loss')
    # plt.yscale('log')
    # plt.legend()
    # plt.title('Loss over 5 epochs')
    # plt.show()


if __name__ == '__main__':
    hw02_code()
