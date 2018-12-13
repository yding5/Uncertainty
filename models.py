'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from densenet import DenseNet3

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_20BCE_beforeAvg(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_beforeAvg, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(8*8*64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out2 = self.linear2(out.view(out.size(0),-1))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        return out1, out2


class ResNet_20BCE_beforeAvg_weightedSoft(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_beforeAvg_weightedSoft, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(8*8*64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out2 = self.linear2(out.view(out.size(0),-1))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out3 = torch.mul(out1,torch.ones_like(out2)-F.sigmoid(out2))
        return out1, out2, out3

class ResNet_20BCE_branch2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_branch2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)

        self.in_planes = 32
        self.N_layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.N_linear1 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out_branch2 = self.layer2(out)
        out = self.layer3(out_branch2)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        N_out = self.N_layer3(out_branch2)
        N_out = F.avg_pool2d(N_out, N_out.size()[3])
        N_out = N_out.view(N_out.size(0), -1)
        out2 = self.N_linear1(N_out)
        return out1, out2

class ResNet_20BCE_branch1(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_branch1, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)

        self.in_planes = 16
        self.N_layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.N_layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.N_linear1 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out_branch1 = self.layer1(out)
        out = self.layer2(out_branch1)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        N_out = self.N_layer2(out_branch1)
        N_out = self.N_layer3(N_out)
        N_out = F.avg_pool2d(N_out, N_out.size()[3])
        N_out = N_out.view(N_out.size(0), -1)
        out2 = self.N_linear1(N_out)
        return out1, out2

class ResNet_decoupled_minus(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_decoupled_minus, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)

        self.in_planes = 16
        self.N_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.N_bn1 = nn.BatchNorm2d(16)
        self.N_layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.N_layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.N_layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.N_linear1 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        N_out = F.relu(self.N_bn1(self.N_conv1(x)))
        N_out = self.N_layer1(N_out)
        N_out = self.N_layer2(N_out)
        N_out = self.N_layer3(N_out)
        N_out = F.avg_pool2d(N_out, N_out.size()[3])
        N_out = N_out.view(N_out.size(0), -1)
        out2 = self.N_linear1(N_out)
        return out1, out2


class ResNet_decoupled_weightedsoft2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_decoupled_weightedsoft2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)

        self.in_planes = 16
        self.N_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.N_bn1 = nn.BatchNorm2d(16)
        self.N_layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.N_layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.N_layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.N_linear1 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        N_out = F.relu(self.N_bn1(self.N_conv1(x)))
        N_out = self.N_layer1(N_out)
        N_out = self.N_layer2(N_out)
        N_out = self.N_layer3(N_out)
        N_out = F.avg_pool2d(N_out, N_out.size()[3])
        N_out = N_out.view(N_out.size(0), -1)
        out2 = self.N_linear1(N_out)
        #print(out1)
        out3 = logweightedsoftmax2(out1,torch.ones_like(out2)-F.sigmoid(out2))
        #print(out3)
        #out3 = torch.mul(out1,torch.ones_like(out2)-F.sigmoid(out2))
        return out1, out2, out3

class ResNet_20BCE_beforeAvg_weightedSoft2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_beforeAvg_weightedSoft2, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(8*8*64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out2 = self.linear2(out.view(out.size(0),-1))
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        #print(out1)
        out3 = logweightedsoftmax2(out1,torch.ones_like(out2)-F.sigmoid(out2))
        #print(out3)
        #out3 = torch.mul(out1,torch.ones_like(out2)-F.sigmoid(out2))
        return out1, out2, out3

def logweightedsoftmax2(x,w):
    #print(x)
    #print(w)
    return x + torch.log(w+0.2) - torch.log(torch.sum(torch.exp(x)*w,1,keepdim=True))

class ResNet_20BCE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        return out1, out2



class ResNet_20BCE_jointforpred(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_jointforpred, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(64, num_classes)
        self.linear3 = nn.Linear(num_classes*2, 64)
        self.linear4 = nn.Linear(64, 10)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        out = self.linear3(torch.cat((out1,out2),1))
        out = F.relu(out)
        out3 = self.linear4(out)
        #out3 = F.sigmoid(out)
        return out1, out2, out3

class ResNet_20BCE_joint(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_20BCE_joint, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear1 = nn.Linear(64, num_classes)
        self.linear2 = nn.Linear(64, num_classes)
        self.linear3 = nn.Linear(num_classes*2, 64)
        self.linear4 = nn.Linear(64, 1)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out1 = self.linear1(out)
        out2 = self.linear2(out)
        out = self.linear3(torch.cat((out1,out2),1))
        out = F.relu(out)
        out3 = self.linear4(out)
        #out3 = F.sigmoid(out)
        return out1, out2, out3

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet20_bce():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet20_bce_neg():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet20_20bce():
    return ResNet_20BCE(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_branch1():
    return ResNet_20BCE_branch1(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_branch2():
    return ResNet_20BCE_branch2(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_joint():
    return ResNet_20BCE_joint(BasicBlock, [3, 3, 3], 10)

def resnet20_decoupled_weightedsoft2():
    return ResNet_decoupled_weightedsoft2(BasicBlock, [3, 3, 3], 10)

def resnet20_decoupled_minus():
    return ResNet_decoupled_minus(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_jointforpred():
    return ResNet_20BCE_jointforpred(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_beforeavg_weightedsoft():
    return ResNet_20BCE_beforeAvg_weightedSoft(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_beforeavg_weightedsoft2():
    return ResNet_20BCE_beforeAvg_weightedSoft2(BasicBlock, [3, 3, 3], 10)

def resnet20_20bce_beforeavg():
    return ResNet_20BCE_beforeAvg(BasicBlock, [3, 3, 3], 10)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])

def resnet38():
    return ResNet(BasicBlock, [6, 6, 6])

def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

def densenet(num_classes=10):
    return DenseNet3(100, num_classes)

def densenet_bce(num_classes=10):
    return DenseNet3(100, num_classes)



def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
