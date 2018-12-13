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
parser.add_argument('--dataset', default='cifar10', type=str, help='training dataset')
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
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        print("undefined num_classes")

    model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes))
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
    train_loader, val_loader = get_data_loader(args.dataset)
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
        target2 = torch.ones((target.shape[0],10))
        target5 = -1*torch.ones((target.shape[0],10))
        target6 = torch.zeros((target.shape[0],10))
        target5 = -1*torch.ones((target.shape[0],10))
        for j, val in enumerate(target):
            target2[j][target[j]]=0
            target5[j][target[j]]=1
            target6[j][target[j]]=1


        target = target.cuda(async=True)
        target2 = target2.cuda(async=True)
        target5 = target5.cuda(async=True)
        target6 = target6.cuda(async=True)
        target5 = target5.cuda(async=True)
        target_var_2 = torch.autograd.Variable(target2)
        target_var_5 = torch.autograd.Variable(target5)
        target_var_6 = torch.autograd.Variable(target6)


        if save:
            targetList = target.tolist()
            for t in targetList:
                res.append([t,])

        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        if args.arch in ['resnet20_20bce_beforeavg','resnet20_20bce','resnet20_decoupled_minus','resnet20_20bce_branch1','resnet20_20bce_branch2']:
            output1, output2 = model(input_var)
            if save:
                output1List = output1.tolist()
                output2List = output2.tolist()
                for out1,out2 in zip(output1List, output2List):
                    res[counter].extend(out1)
                    res[counter].extend(out2)
                    counter = counter + 1

        elif args.arch in ['resnet20_20bce_joint','resnet20_20bce_beforeavg_weightedsoft','resnet20_20bce_beforeavg_weightedsoft2','resnet20_20bce_jointforpred','resnet20_decoupled_weightedsoft2']:
            output1, output2, output3 = model(input_var)
            if save:
                output1List = output1.tolist()
                output2List = output2.tolist()
                output3List = output3.tolist()
                for out1,out2,out3 in zip(output1List, output2List, output3List):
                    res[counter].extend(out1)
                    res[counter].extend(out2)
                    res[counter].extend(out3)
                    counter = counter + 1
        elif args.arch in ['resnet20_bce','resnet20_bce_neg']:
            output2 = model(input_var)
            if save:
                output2List = output2.tolist()
                for out2 in output2List:
                    res[counter].extend(out2)
                    counter = counter + 1
        else:
            output1 = model(input_var)
            if save:
                output1List = output1.tolist()
                for out1 in output1List:
                    res[counter].extend(out1)
                    counter = counter + 1
        #loss1
        if args.arch in ['resnet20_bce']:
            loss1 = criterionList[1](output2, target_var_2)
        elif args.arch in ['resnet20_bce_neg']:
            loss1 = criterionList[6](output2, target_var_6)
        else:
            loss1 = criterionList[0](output1, target_var)



        # measure accuracy and record loss
        if args.arch not in ['resnet20_20bce_jointforpred','resnet20_decoupled_minus','resnet20_20bce_branch1','resnet20_20bce_branch2','resnet20_20bce','resnet44','resnet38','resnet20_bce_neg']:
            output3 = output3.float()
            loss1 = loss1.float()
            losses.update(loss1.data[0], input.size(0))
            prec1 = accuracy(output3.data, target)[0]
            top1.update(prec1[0], input.size(0))
            prec2 = accuracy2(output2.data, target)[0]
            top1FromNot.update(prec2[0], input.size(0))
        if args.arch in ['resnet20_decoupled_weightedsoft2']:
            prec2 = accuracy2(output2.data, target)[0]
            top1FromNot.update(prec2[0], input.size(0))
            output3 = output3.float()
            prec3 = accuracy(output3.data, target)[0]
            top1Comb.update(prec3[0], input.size(0))
        if args.arch not in ['resnet20_bce','resnet20_bce_neg']:
            output1 = output1.float()
            loss1 = loss1.float()
            losses.update(loss1.data[0], input.size(0))
            prec1 = accuracy(output1.data, target)[0]
            top1.update(prec1[0], input.size(0))
        if args.arch in ['resnet20_bce_neg']:
            output2 = output2.float()
            loss1 = loss1.float()
            losses.update(loss1.data[0], input.size(0))
            prec2 = accuracy(output2.data, target)[0]
            top1.update(prec2[0], input.size(0))
        
        if args.arch in ['resnet20_20bce_beforeavg','resnet20_20bce','resnet20_20bce_joint','resnet20_20bce_beforeavg_weightedsoft','resnet20_bce','resnet20_20bce_beforeavg_weightedsoft2','resnet20_decoupled_minus','resnet20_20bce_branch1','resnet20_20bce_branch2']:
            prec2 = accuracy2(output2.data, target)[0]
            top1FromNot.update(prec2[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.arch in ['resnet20_20bce_beforeavg','resnet20_20bce','resnet20_20bce_joint','resnet20_20bce_beforeavg_weightedsoft','resnet20_20bce_beforeavg_weightedsoft2','resnet20_20bce_jointforpred','resnet20_decoupled_minus','resnet20_20bce_branch1','resnet20_20bce_branch2','resnet20_bce_neg']:
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1FroNot {top1FromNot.val:.4f} ({top1FromNot.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, top1FromNot=top1FromNot,
                      top1=top1))
        elif args.arch in ['resnet20_bce']:
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1FroNot {top1FromNot.val:.4f} ({top1FromNot.avg:.4f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, top1FromNot=top1FromNot))
        elif args.arch in ['resnet20_decoupled_weightedsoft2']:
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1FroNot {top1FromNot.val:.4f} ({top1FromNot.avg:.4f})\t'
                  'Prec@1Comb {top1Comb.val:.4f} ({top1Comb.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses, top1FromNot=top1FromNot,top1Comb=top1Comb,top1=top1))
        else:
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))


    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    npres = np.asarray(res)
    np.save(args.arch+str(args.weightForNot)+str(args.weightForSoft)+str(args.weightForWeightedSoft)+'res.npy', npres)



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
