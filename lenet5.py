import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from utils import Cutout, CIFAR10Policy, evaluate_accuracy
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import random

def seed_all(seed=1000):        #设置随机种子，保证可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_act = []


def hook(module, input, output):        #用于记录每层的输出数值
    sort, _ = torch.sort(output.detach().view(-1).cpu())
    max_act.append(sort[int(sort.shape[0] * 0.99) - 1])     #升序后，max_act存储了该层输出中占据99%的那个值


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        cnn = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.BatchNorm2d(120),
            nn.ReLU(),
            
        )
        hooks = []
        self.conv = cnn     #卷积层
        self.fc = nn.Linear(120, 10, bias=True)
        for i in range(len(self.conv)):
            hooks.append(self.conv[i].register_forward_hook(hook))
        hooks.append(self.fc.register_forward_hook(hook))
        self.hooks = hooks

    def forward(self, img):
        conv = self.conv(img)
        x = conv.view(conv.shape[0], -1)
        output = self.fc(x)
        return output




def train(net,train_iter,test_iter,optimizer,scheduler,device,num_epochs):
    best = 0
    net = net.to(device)
    print("training on ",device)
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        
        losses = []
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            label = y
            l = loss(y_hat,label)
            losses.append(l.cpu().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter,net)
        losses.append(np.mean(losses))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), './MNIST_max.pth')


if __name__ == '__main__':
    seed_all(42)
    data_train = MNIST('./data/mnist',
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    data_test = MNIST('./data/mnist',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))
    train_iter = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    test_iter = DataLoader(data_test, batch_size=1024, num_workers=8)

    lr,num_epochs = 0.1, 5
    net = LeNet5()
    [net.hooks[i].remove() for i in range(len(net.hooks))]  #移除了模型中之前定义的所有钩子函数，确保在训练之前没有残留的钩子
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs)

    net.load_state_dict(torch.load("./MNIST_max.pth"))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter,net,device)
    print(acc)
