from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from numpy import arange

import global_vars

schedulers = []
conds = []


def getDefaultScheduler(optimizer):
    return MultiStepLR(optimizer=optimizer, milestones=arange(30, 100), gamma=0.1 ** (1 / 70))


def getDefaultSchedulerShifted2(optimizer):
    return MultiStepLR(optimizer=optimizer, milestones=arange(30, 150), gamma=0.3 ** (1 / 120))


def getDefaultSchedulerShifted(optimizer):
    return MultiStepLR(optimizer=optimizer, milestones=arange(80, 170), gamma=0.1 ** (1 / 70))


def getLongerDecayScheduler(optimizer):
    return LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=120, last_epoch=-1)


def getDefaultWarmUpScheduler(optimizer):
    return LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=1.0, total_iters=10, last_epoch=-1)


def getLongerWarmUpScheduler(optimizer):
    return LinearLR(optimizer=optimizer, start_factor=0.001, end_factor=1.0, total_iters=50, last_epoch=-1)


def setSchedulers(optimizer):
    if global_vars.args.scheduler_name == "default":
        schedulers.append(getDefaultWarmUpScheduler(optimizer))

        def cond(epoch):
            return epoch < 10

        conds.append(cond)
        schedulers.append(getDefaultScheduler(optimizer))

        def cond2(epoch):
            return True

        conds.append(cond2)
    elif global_vars.args.scheduler_name == "defaultwithlongerdecay":
        schedulers.append(getDefaultWarmUpScheduler(optimizer))

        def cond(epoch):
            return epoch < 10

        conds.append(cond)
        schedulers.append(getLongerDecayScheduler(optimizer))

        def cond2(epoch):
            return epoch >= 30 and epoch < 150

        conds.append(cond2)
    elif global_vars.args.scheduler_name == "defaultwithlongerdecay2":
        schedulers.append(getDefaultWarmUpScheduler(optimizer))

        def cond(epoch):
            return epoch < 10

        conds.append(cond)
        schedulers.append(getDefaultSchedulerShifted2(optimizer))

        def cond2(epoch):
            return True

        conds.append(cond2)
    elif global_vars.args.scheduler_name == "defaultwithlongerwarmup":
        schedulers.append(getLongerWarmUpScheduler(optimizer))

        def cond(epoch):
            return epoch < 50

        conds.append(cond)
        schedulers.append(getDefaultSchedulerShifted(optimizer))

        def cond2(epoch):
            return True

        conds.append(cond2)
    else:
        raise Exception("Scheduler not implemented.")


def schedulersStep(epoch):
    for scheduler, cond in zip(schedulers, conds):
        if (cond(epoch) and epoch < 50):
            scheduler.step()
