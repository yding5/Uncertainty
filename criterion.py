import torch
import torch.nn as nn



def get_criterion_list(name):
    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.111 for i in range(10)])).cuda()
    criterion3 = nn.MSELoss().cuda()
    criterion4 = nn.CrossEntropyLoss().cuda()
    criterion5 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.05 for i in range(10)])).cuda()
    criterion6 = nn.NLLLoss().cuda()
    criterion7 = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([9.0 for i in range(10)])).cuda()
    criterionList = [criterion,criterion2,criterion3,criterion4,criterion5,criterion6,criterion7]
    return criterionList
