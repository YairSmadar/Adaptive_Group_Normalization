import os
# import shutil
# import unittest
import copy
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_vars
from resnet import resnet50
from scedulers import setSchedulers, schedulersStep
from data_loading import getLoaders

import torch
from torch.cuda.random import initial_seed
import torch.nn as nn

def main():
    global best_prec1
    args = global_vars.args

    # print parameters
    global_vars.printParameters()

    # create model
    print("=> creating model resnet50")
    model = resnet50()
    model = torch.nn.DataParallel(model).to(global_vars.device)
    
    criterion = nn.CrossEntropyLoss().to(global_vars.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # initialize the results arrays
    lr_array = []
    # get schedulers
    setSchedulers(optimizer)

    for epoch in range(args.start_epoch, args.epochs):
        optimizer.zero_grad()
        optimizer.step()

        # adjust the learning rate
        schedulersStep(epoch)

        lr_array.append(optimizer.param_groups[0]['lr'])
    
    plt.plot(lr_array)
    plt.ylabel('lr')
    # plt.yscale('log')
    plt.xlabel('epoch')
    plt.savefig('/content/lr_scheduler.png')
    plt.show()
    plt.close()




if __name__ == '__main__':
    main()