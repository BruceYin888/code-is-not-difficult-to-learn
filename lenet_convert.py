import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import time
from lenet5 import evaluate_accuracy

max_act = []
gamma = 2


def hook(module, input, output):
    '''
    use hook to easily get the maximum of each layers based on one training batch
    '''
    out = output.detach()
    out[out>1] /= gamma
    sort, _ = torch.sort(out.view(-1).cpu())
    max_act.append(sort[int(sort.shape[0] * 0.999) - 1])

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

#=========== Do weight Norm, replace ReLU with SpikeNode =============#
class SpikeNode(nn.Module):
    def __init__(self, smode=True, gamma=5):
        super(SpikeNode, self).__init__()
        self.smode = smode
        self.mem = 0
        self.spike = 0
        self.sum = 0
        self.threshold = 1.0
        self.opration = nn.ReLU(True)
        self.rsum = []
        self.summem = 0
        self.rmem = []
        self.gamma = gamma
    def forward(self,x):
        if not self.smode:
            out = self.opration(x)
        else:
            self.mem = self.mem + x

            self.spike = (self.mem / self.threshold).floor().clamp(min=0, max=self.gamma)
            self.mem = self.mem - self.spike
            out = self.spike
        return out

class SMaxPool(nn.Module):
    def __init__(self, smode=True, lateral_inhi=False):
        super(SMaxPool, self).__init__()
        self.smode = smode      #是否处于脉冲模式
        self.lateral_inhi = lateral_inhi    #是否启用横向抑制机制
        self.sumspike = None        #用于在脉冲模式下累积输入的变量
        self.opration = nn.MaxPool2d(kernel_size=2, stride=2)   #最大池化操作
        self.sum = 0
        self.input = 0

    def forward(self, x):
        if not self.smode:      #不处于脉冲模式
            out = self.opration(x)  #最大池化操作
        elif not self.lateral_inhi: 
            self.sumspike += x  #通过累积输入来模拟脉冲神经元的行为
            single = self.opration(self.sumspike * 1000)   #对累积的输入进行最大池化 
            sum_plus_spike = self.opration(x + self.sumspike * 1000)    #将输入和累积的输入相加并进行最大池化
            out = sum_plus_spike - single   #计算最终的输出
        else:   
            self.sumspike += x
            out = self.opration(self.sumspike)
            self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest') #是在脉冲模式下，通过横向抑制的方式，减少 self.sumspike 中的信息，以模拟脉冲神经元之间的相互影响。
        return out
    
def fuse_norm_replace(m, max_activation, last_max, smode=True, gamma=5, data_norm=True, lateral_inhi=False):    #合并卷积和批标准化层，并进行数据归一化
    '''
    merge conv and bn, then do data_norm
    :param m:                model
    :param max_activation:   the max_activation values on one training batch
    :param last_max:         the last max
    :param smode:            choose to use spike
    :param data_norm:
    :param lateral_inhi:
    :return:                 snn
    '''
    global index
    children = list(m.named_children())
    c, cn = None, None

    for i, (name, child) in enumerate(children):
        ind = index
        if isinstance(child, nn.Linear):    #若子模块是线性层，根据 data_norm 参数对权重和偏置进行归一化
            if data_norm:
                print("index:", index)
                print("max_activation length:", len(max_activation))

                child.weight.data /= max_activation[index] / max_activation[index-2]
                child.bias.data /= max_activation[index]
                last_max = max_activation[index]
        elif isinstance(child, nn.BatchNorm2d): #如果是二维批标准化层 (nn.BatchNorm2d)，通过 fuse 函数融合卷积层和批标准化层，然后对权重和偏置进行归一化
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = torch.nn.Identity()
            if data_norm:
                print("index:", index)
                print("max_activation length:", len(max_activation))

                m._modules[cn].weight.data /= max_activation[index] / last_max
                m._modules[cn].bias.data /= max_activation[index]
                last_max = max_activation[index]
            c = None
        elif isinstance(child, nn.Conv2d):  #如果是卷积层(nn.Conv2d)，记录该层为当前卷积层。
            c = child
            cn = name
        elif isinstance(child, nn.ReLU):    #如果是ReLU激活层，替换为脉冲神经元层 SpikeNode
            m._modules[name] = SpikeNode(smode=smode, gamma=gamma)
            if not data_norm:
                m._modules[name].threshold = max_activation[index]
                last_max = max_activation[index]
        elif isinstance(child,nn.LogSoftmax):   #将LogSoftmax激活曾替换为SpikeNode
            m._modules[name] = SpikeNode(smode=smode, gamma=gamma)
            if not data_norm:
                m._modules[name].threshold = max_activation[index]
                last_max = max_activation[index]
        elif isinstance(child, nn.MaxPool2d):   #如果是最大池化层(nn.MaxPool2d)，替换为脉冲神经元最大池化层 SMaxPool
            m._modules[name] = SMaxPool(smode=smode, lateral_inhi=lateral_inhi)
        elif isinstance(child, nn.AvgPool2d):   #如果是平均池化层，忽略
            pass
        else:       #其他子模块，递归调用fuse_norm_replace
            fuse_norm_replace(child, max_activation, last_max, smode, gamma, data_norm, lateral_inhi)
            index -= 1
        index += 1

def fuse(conv, bn): #融合卷积层和批标准化层层
    '''
    fuse the conv and bn layer
    '''
    w = conv.weight
    mean, var_sqrt, beta, gamma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps), bn.weight, bn.bias
    b = conv.bias if conv.bias is not None else mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def clean_mem_spike(m):
    '''
    when change batch, you should clean the mem and spike of last batch
    :param m:  snn
    :return:
    '''
    children = list(m.named_children())
    for name, child in children:
        if isinstance(child, SpikeNode):
            child.mem = 0
            child.spike = 0
        elif isinstance(child, SMaxPool):
            child.sumspike = 0
        else:
            clean_mem_spike(child)

def evaluate_snn(test_iter, snn, net, device=None, duration=50, plot=False, linetype=None):
    linetype = '-' if linetype==None else linetype
    accs = []
    acc_sum, n = 0.0, 0
    snn.eval()

    for test_x, test_y in tqdm(test_iter):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            clean_mem_spike(snn)
            acc = []
            for t in range(duration):
                start = time.time()
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
        accs.append(np.array(acc))

    accs = np.array(accs).mean(axis=0)

    print(max(accs))
    if plot:
        plt.plot(list(range(len(accs))), accs, linetype)
        plt.ylabel('Accuracy')
        plt.xlabel('Time Step')
        # plt.show()
        plt.savefig('./result.jpg')

if __name__ == '__main__':
    global index
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
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

    #the result of ANN
    net = LeNet5()
    net1 = deepcopy(net)
    [net1.hooks[i].remove() for i in range(len(net1.hooks))]
    net1.load_state_dict(torch.load("./MNIST_max.pth",map_location=torch.device(device)))
    net1 = net1.to(device)
    acc = evaluate_accuracy(test_iter, net1, device)
    print("acc on ann is : {:.4f}".format(acc))

    # get max activation on one training batch
    net2 = deepcopy(net)
    net2.load_state_dict(torch.load("./MNIST_max.pth", map_location=torch.device(device)))
    net2 = net2.to(device)
    _ = evaluate_accuracy(train_iter, net2, device, only_onebatch=True)
    [net2.hooks[i].remove() for i in range(len(net2.hooks))]

    # data_norm
    net3 = deepcopy(net2)
    index = 0
    fuse_norm_replace(net3, max_act, last_max=1.0, smode=False, data_norm=True)

    index = 0
    fuse_norm_replace(net2, max_act, last_max=1.0, smode=True, gamma=gamma, data_norm=True, lateral_inhi=False)
    evaluate_snn(test_iter, net2, net3, device=device, duration=256, plot=True, linetype=None)