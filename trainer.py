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
from dataloader import get_data_loader
from criterion import get_criterion_list
from utilty import get_loss_and_accuracy
from utilty import get_lr_scheduler


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument ('--size_dense', default=100, type=int,
                    help='depth of densenet')
parser.add_argument ('--size_wide', default=10, type=int,
                    help='widden facor of wideresnet')
parser.add_argument('--softmax_threshold_value', default=0.95, type=float, help='threshold value for the softmax')
parser.add_argument('--softmax_threshold', default=False, type=bool, help='use of softmax threshold')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--nesterov', default=False, type=bool, help='use of nesterov momentum')
parser.add_argument('--normalize_loss_weight', default=0, type=int, help='type of normalized weight of positive samples')
parser.add_argument('--binarized_label', default=10, type=int, help='type of binarized_label 10 is equal to no binarized label')
parser.add_argument('--optimizer', default='SGD', type=str, help='use of optimizer from SGD and ADAM')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_schedule', default='default', type=str, help='learning rate schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--weightForNot', default=0.5, type=float,
                    metavar='W', help='weight for Not class in total loss')
parser.add_argument('--weightForSoft', default=1.0, type=float,
                    metavar='W', help='weight for regular softmax class in total loss')
parser.add_argument('--weightForWeightedSoft', default=5.0, type=float,
                    metavar='W', help='weight for regular softmax class in total loss')
parser.add_argument('--weightForConf', default=0.1, type=float,
                    metavar='W', help='weight for confidence estimation in total loss')
parser.add_argument('--weightForJointPred', default=2.0, type=float,
                    metavar='W', help='weight for confidence estimation in total loss')
parser.add_argument('--weightForMinus', default=0.4, type=float,
                    metavar='W', help='weight for confidence estimation in total loss')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #Some hard-code setting for fast experiments
    if args.dataset in ['cifar10','cifar10_extra']:
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        print("undefined num_classes")

    if args.arch.startswith('densenet'):
        args.epochs = 300
        print('set epochs as 300 automatically')
        #args.batch_size = 64
        #print('set batch size as 64 automatically')
        args.nesterov = True
        print('set nesterov as True automatically')
    elif args.arch.startswith('resnet'):
        args.epochs = 200
        print('set epochs as 200 automatically')
        args.batch_size = 128
        print('set batch size as 128 automatically')
    elif args.arch.startswith('wideresnet'):
        args.epochs = 200
        print('set epochs as 200 automatically')
        args.batch_size = 128
        print('set batch size as 128 automatically')
    elif args.arch.startswith('vgg'):
        args.epochs = 45
        print('set epochs as 45 automatically')
        args.batch_size = 128
        print('set batch size as 128 automatically')
        args.lr = 0.01
        print('set lr as 1e-2 automatically')
    else:
        print("undefined epochs")


    if args.arch.startswith('dense'):
        size = args.size_dense
    elif args.arch.startswith('wide'):
        size = args.size_wide
    elif args.arch.startswith('vgg'):
        size = 0
    else:
        raise NotImplementedError
    model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes,size))
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #Get data
    train_loader, val_loader = get_data_loader(args.dataset, args.batch_size)
    # define loss function (criterion) and optimizer
    criterionList = get_criterion_list(args.arch)

    ####################################################
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

    lr_scheduler = get_lr_scheduler(optimizer,args)
    #if args.arch.startswith('vgg'):
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
     #                                                   milestones=[int(args.epochs/2), int(args.epochs/4*3)], last_epoch=args.start_epoch - 1)


    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterionList, optimizer, epoch, args, num_classes, args.binarized_label, args.softmax_threshold, args.softmax_threshold_value)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterionList, args, best_prec1, num_classes)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if prec1 == best_prec1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'best.th'))

        print("current best: {}".format(best_prec1))

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, filename=os.path.join(args.save_dir, 'final.th'))


def train(train_loader, model, criterionList, optimizer, epoch, args, num_classes, binarized_label, softmax_threshold, softmax_threshold_value):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1FromNot = AverageMeter()
    top1Comb = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
    
        lossList, accuracyList, _ = get_loss_and_accuracy(args.arch, model, input, target, num_classes, args.normalize_loss_weight, binarized_label, softmax_threshold, softmax_threshold_value)

        # measure data loading time
        data_time.update(time.time() - end)

        loss = lossList[0]
        accuracy = accuracyList[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.float()
        losses.update(loss.data[0], input.size(0))
        top1.update(accuracy[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, model, criterionList, args, best_prec1, num_classes):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    top1FromNot = AverageMeter()
    top1Comb = AverageMeter()

    # switch to evaluate mode
    model.eval()


    for i, (input, target) in enumerate(val_loader):
        lossList, accuracyList, _ = get_loss_and_accuracy(args.arch, model, input, target, num_classes, args.normalize_loss_weight)

        loss = lossList[0]
        accuracy = accuracyList[0]

        loss = loss.float()
        losses.update(loss.data[0], input.size(0))
        top1.update(accuracy[0], input.size(0))

        if i % args.print_freq == 0:
           print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses,
                      top1=top1))


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))


    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy2(output, target, topk=(1,)):
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


if __name__ == '__main__':
    main()
