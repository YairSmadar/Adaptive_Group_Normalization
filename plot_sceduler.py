import matplotlib.pyplot as plt
import global_vars
from resnet import resnet50
from scedulers import SchedulerManager

import torch
import torch.nn as nn


def main():
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
    scheduler_manager = SchedulerManager(args.scheduler_name)
    scheduler_manager.set_schedulers(optimizer)

    for epoch in range(args.start_epoch, args.epochs):
        optimizer.zero_grad()
        optimizer.step()

        # adjust the learning rate
        scheduler_manager.schedulers_step(epoch)

        lr_array.append(optimizer.param_groups[0]['lr'])
    
    plt.plot(lr_array)
    plt.ylabel('lr')
    plt.xlabel('epoch')
    plt.savefig(f'res\\{scheduler_manager.get_scheduler_name()}_scheduler.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()