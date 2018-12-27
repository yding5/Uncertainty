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


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--train_dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--infer_dataset', default='cifar10', type=str, help='training dataset')
parser.add_argument('--filename', default='name_not_specified', type=str, help='use of nesterov momentum')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
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
parser.add_argument('--print-freq', '-p', default=1000, type=int,
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
    if args.train_dataset == 'cifar10':
        num_classes = 10
    elif args.train_dataset == 'cifar100':
        num_classes = 100
    else:
        print("undefined num_classes")

    #model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes))
    #model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            #checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            #model.load_state_dict(checkpoint['state_dict'])
            model = torch.load(args.resume)
            model = model.cuda()
            #print("=> loaded checkpoint '{}' (epoch {})"
                  #.format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #Get data
    train_loader, val_loader = get_data_loader(args.infer_dataset)
    # define loss function (criterion) and optimizer
    criterionList = get_criterion_list(args.arch)




    validate_and_save(val_loader, model, criterionList, args, num_classes)


def validate_and_save(val_loader, model, criterionList, args, num_classes):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1FromNot = AverageMeter()
    top1Comb = AverageMeter()

    # switch to evaluate mode
    model.eval()

    res = []
    counter = 0

    end = time.time()
    for i, (input, target) in enumerate(val_loader):



        lossList, accuracyList, outputList = get_loss_and_accuracy(args.arch, model, input, target, num_classes)

        loss = lossList[0]
        accuracy = accuracyList[0]

        loss = loss.float()
        losses.update(loss.data[0], input.size(0))
        top1.update(accuracy[0], input.size(0))

        targetList = target.tolist()
        concatenated = torch.cat(outputList, dim=1).tolist()
        #print(len(concatenated))
        #print(len(concatenated[0]))
        for t, o in zip(targetList, concatenated):
            #print(o)
            #print(len(o))
            tempList = [t,]
            tempList.extend(o)
            res.append(tempList)

        if i % args.print_freq == 0:
           print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses,
                      top1=top1))


    npres = np.asarray(res)
    np.save(args.filename+'.npy', npres)

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


if __name__ == '__main__':
    main()
