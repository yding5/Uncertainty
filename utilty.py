import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import numpy as np



def get_loss_and_accuracy(name, model, input, target, num_classes):
    lossList = []
    accuracyList = []
    if name == 'densenet':
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input).cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        outputList = model(input_var)
        lossList.append( criterion(outputList[0], target_var) )
        accuracyList.append( accuracy(outputList[0].data, target) )
    elif name == 'densenet_bce':
        target2 = torch.ones((target.shape[0], num_classes))
        for j, val in enumerate(target):
            target2[j][target[j]]=0

        target = target.cuda(async=True)
        target2 = target2.cuda(async=True)
        target_var_2 = torch.autograd.Variable(target2)
        input_var = torch.autograd.Variable(input).cuda()

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.111 for i in range(num_classes)])).cuda()
        outputList = model(input_var)
        lossList.append( criterion(outputList[0], target_var_2) )
        accuracyList.append( accuracy_bottom(outputList[0].data, target) )



    else:
        print("undefined loss and accuracy for {}!".format(name))
        return -1

    return lossList, accuracyList, outputList

    #criterionCE = nn.CrossEntropyLoss().cuda()
    #criterionBCE = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.111 for i in range(10)])).cuda()
    #criterion3 = nn.MSELoss().cuda()
    #criterion4 = nn.CrossEntropyLoss().cuda()
    #criterion5 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.05 for i in range(10)])).cuda()
    #criterion6 = nn.NLLLoss().cuda()
    #criterion7 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([9.0 for i in range(10)])).cuda()
    #criterionList = [criterion,criterion2,criterion3,criterion4,criterion5,criterion6,criterion7]









def accuracy_bottom(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, False, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

