import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import numpy as np

class customizedLR(torch.optim.lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones. When
    last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, last_epoch=-1):
        super(customizedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= 14:
            return [base_lr * (1+ self.last_epoch/14*9)
                for base_lr in self.base_lrs]
        elif self.last_epoch <= 28:
            return [base_lr * (10 - (self.last_epoch-14)/14*9)
                for base_lr in self.base_lrs]
        else:
            return [base_lr * (1- (self.last_epoch-28)/16*0.999)
                for base_lr in self.base_lrs]

def get_lr_scheduler(optimizer, args):
    if args.arch.startswith('vgg'):
        lr_scheduler = customizedLR(optimizer,last_epoch=args.start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[int(args.epochs/2), int(args.epochs/4*3)], last_epoch=args.start_epoch - 1) 
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    return lr_scheduler 

    


def get_loss_and_accuracy(name, model, input, target, num_classes, normalize_loss_weight, binarized_label=10):
    lossList = []
    accuracyList = []

    if binarized_label < 10:
        for j, val in enumerate(target):
            if val == binarized_label:
                target[j]=1
            else:
                target[j]=0
        

    if name in ['densenet','vgg16','vgg16_bn','wideresnet']:
        target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input).cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        outputList = model(input_var)
        lossList.append( criterion(outputList[0], target_var) )
        accuracyList.append( accuracy(outputList[0].data, target) )
    elif name in ['densenet_bce','vgg16_bce','vgg16_bn_bce','wideresnet_bce','densenet_partialsharing1_bce']:
        target2 = torch.ones((target.shape[0], num_classes))
        for j, val in enumerate(target):
            target2[j][target[j]]=0

        target = target.cuda(async=True)
        target2 = target2.cuda(async=True)
        target_var_2 = torch.autograd.Variable(target2)
        input_var = torch.autograd.Variable(input).cuda()

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.111 for i in range(num_classes)])).cuda()
        outputList = model(input_var)
        if normalize_loss_weight == 2:
            lossList.append( 0.5 * criterion(outputList[0], target_var_2) )
        elif normalize_loss_weight == 3:
            lossList.append( 9 * criterion(outputList[0], target_var_2) )
        elif normalize_loss_weight == 4:
            lossList.append( 5 * criterion(outputList[0], target_var_2) )
        else:
            lossList.append( criterion(outputList[0], target_var_2) )
        accuracyList.append( accuracy_bottom(outputList[0].data, target) )

    elif name in ['densenet_bce_neg','vgg16_bce_neg','vgg16_bn_bce_neg','wideresnet_bce_neg','densenet_partialsharing1_bce_neg']:
        target2 = torch.zeros((target.shape[0], num_classes))
        for j, val in enumerate(target):
            target2[j][target[j]]=1

        target = target.cuda(async=True)
        target2 = target2.cuda(async=True)
        target_var_2 = torch.autograd.Variable(target2)
        #target_var = torch.autograd.Variable(target)
        input_var = torch.autograd.Variable(input).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([9 for i in range(num_classes)])).cuda()
        outputList = model(input_var)
        if normalize_loss_weight == 1:
            lossList.append( 0.111 * criterion(outputList[0], target_var_2) )
        elif normalize_loss_weight == 2:
            lossList.append( 0.111 * 0.5 * criterion(outputList[0], target_var_2) )
        elif normalize_loss_weight == 3:
            lossList.append( 5/9 * criterion(outputList[0], target_var_2) )
        else:
            lossList.append( criterion(outputList[0], target_var_2) )
        accuracyList.append( accuracy(outputList[0].data, target) )



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

