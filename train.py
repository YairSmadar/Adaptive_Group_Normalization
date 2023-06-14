import global_vars

if global_vars.args.use_wandb:
    import wandb
from copy import deepcopy
from os.path import isfile
from time import time
from numpy.random import seed as seed1
from resnet import resnet50
from scedulers import SchedulerManager
from data_loading import getLoaders

from torch import use_deterministic_algorithms, load, no_grad
from torch.random import manual_seed as seed2
from torch.cuda import is_available, manual_seed as seed3
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
import torch

seed1(global_vars.args.seed)
seed2(global_vars.args.seed)
if is_available():
    seed3(global_vars.args.seed)
deterministic = True
benchmark = False
use_deterministic_algorithms(True)


def main():

    args = global_vars.args

    if global_vars.args.use_wandb:
        wandb.init(project="Adaptive_Normalization",
                   entity="the-smadars",
                   name=global_vars.generate_wandb_name(),
                   config=args)
        wandb.run.summary["best_test_accuracy"] = 0
        wandb.run.summary["best_test_loss"] = 999

    # print parameters
    global_vars.printParameters()

    # create model
    print("=> creating model resnet50")
    model = resnet50()
    model = DataParallel(model).to(global_vars.device)

    # define loss function (criterion) and optimizer
    criterion = CrossEntropyLoss().to(global_vars.device)
    optimizer = SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # CIFAR100 data loading code
    train_loader, val_loader, reclustring_loader = getLoaders(args.dataset, global_vars.generator)

    if args.resume:
        # optionally resume from a checkpoint
        if isfile(args.resume):
            print("=> loading checkpoint from '{}'".format(global_vars.args.resume))
            model, _optimizer = global_vars.load_checkpoint(model, optimizer)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # load & save the model initial weights 
        if args.load != None:
            print("loading weights from:", args.load)
            State_dict = load(args.load, map_location=torch.device(global_vars.device_name))['state_dict']
            with no_grad():
                # Only update the weights if they are in both the
                # model's and the loaded state dict
                for param1 in model.state_dict():
                    if param1 in State_dict:
                        model.state_dict()[param1].data.fill_(0).add_(
                            State_dict[param1].data)
        global_vars.zero_state_dict = deepcopy(model.state_dict())
        global_vars.save_initial_weights()

    if args.evaluate:
        print("evaluate:")
        validate(val_loader, model, criterion, 0)
        return

    global_vars.init_saveing_path()

    # set schedulers
    scheduler_manager = SchedulerManager(args.scheduler_name, args.epochs)
    scheduler_manager.set_schedulers(optimizer)

    # get the learning rate to the starting value
    for epoch in range(0, args.start_epoch):
        # adjust optimizer with no loss
        optimizer.zero_grad()
        optimizer.step()

        # adjust the learning rate
        scheduler_manager.schedulers_step(epoch)

    for epoch in range(args.epochs):

        global_vars.epoch_num = epoch

        #  When loading model form check-point - iterate
        #  the data in order to set the data to be exact as the last checkpoint
        get_to_start_epoch = False
        if epoch < args.start_epoch:
            get_to_start_epoch = True

        # train for one epoch and evaluate on validation set
        train_loss, train_prc1, train_prc5 = train(train_loader, model, criterion, optimizer, epoch, get_to_start_epoch)
        test_loss, test_prc1, test_prc5 = validate(val_loader, model, criterion, epoch, get_to_start_epoch)

        if global_vars.args.use_wandb:
            wandb.log({"train loss": train_loss,
                       "train accuracy (top1)": train_prc1,
                       "test loss": test_loss,
                       "test accuracy": test_prc1
                       })

            wandb.run.summary["best_test_accuracy"] = \
                test_prc1 if test_prc1 > wandb.run.summary["best_test_accuracy"] \
                    else wandb.run.summary["best_test_accuracy"]
            wandb.run.summary["best_test_loss"] = \
                train_loss if train_loss < wandb.run.summary["best_test_loss"] \
                    else wandb.run.summary["best_test_loss"]

        if not get_to_start_epoch:
            # save epoch prcitions and losses
            global_vars.save_results(train_loss, train_prc1, train_prc5, test_loss, test_prc1, test_prc5)

            # adjust the learning rate
            scheduler_manager.schedulers_step(epoch)

            # save checkpoint
            global_vars.save_checkpoint(model, optimizer, epoch)


def train(train_loader, model, criterion, optimizer, epoch, get_to_start_epoch):

    epoch_time = time()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_freq = global_vars.args.print_freq

    # switch to train mode
    model.train()
    global_vars.train_mode = True

    end = time()
    num_of_batches = len(train_loader.batch_sampler)
    for i, (input, target) in enumerate(train_loader):
        
        if input.size()[0] != global_vars.args.batch_size:
            continue

        if not get_to_start_epoch:
            # set the recluster boolean (for RGN & SGN layers).
            if global_vars.is_agn:
                if global_vars.args.cluster_last_batch:
                    global_vars.recluster = (i == num_of_batches - 1)
                else:
                    global_vars.recluster = (i == 0)

                shifted_epoch = epoch + global_vars.args.epoch_start_cluster
                epoch_clustring_loop = shifted_epoch % global_vars.args.norm_shuffle
                if global_vars.recluster:
                    global_vars.recluster = (
                                                        epoch_clustring_loop < global_vars.args.riar) and epoch < global_vars.args.max_norm_shuffle

                if global_vars.args.shuf_each_batch:
                    global_vars.recluster = True

                if global_vars.args.save_shuff_idxs and (epoch_clustring_loop < global_vars.args.riar):
                    global_vars.recluster = True

            # measure data loading time
            data_time.update(time() - end)

            # compute output
            model.module.batch_num = i
            output = model(input)

            if global_vars.device_name == 'cuda':
                target = target.cuda(non_blocking=True)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if i % print_freq == 0:
                print('Train:\t[{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

            # # Optional
            # if global_vars.args.use_wandb:
            #     wandb.watch(model)

    if not get_to_start_epoch:
        print(
            'Train:\t[{0}]\tLoss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'.format(epoch, loss=losses,
                                                                                                        top1=top1,
                                                                                                        top5=top5))

    print(f"Total Epoch {epoch} time: {time() - epoch_time}\n")
    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, epoch, get_to_start_epoch=False):
    eval_time = time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_freq = global_vars.args.print_freq

    global_vars.recluster = False  # no need to recluster in validation
    global_vars.train_mode = False

    # switch to evaluate mode
    model.eval()

    end = time()

    for i, (input, target) in enumerate(val_loader):

        if input.size()[0] != global_vars.args.batch_size:
            continue

        if not get_to_start_epoch:
            if global_vars.device_name == 'cuda':
                target = target.cuda(non_blocking=True)

            # compute output
            with no_grad():
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if i % print_freq == 0:
                print('Test:\t[{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    if not get_to_start_epoch:
        print('Test:\t[{0}]\tLoss {loss.avg:.4f}\tPrec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'.format(epoch, loss=losses,
                                                                                                         top1=top1,
                                                                                                        top5=top5))

    print(f"Total Eval {epoch} time: {time() - eval_time}\n")

    return losses.avg, top1.avg, top5.avg


def channelsReclustering(reclustring_loader, model, criterion, epoch):
    for i, (input, target) in enumerate(reclustring_loader):
        if global_vars.device_name == 'cuda':
            target = target.cuda(non_blocking=True)

        # compute output
        with no_grad():
            output = model(input)
            loss = criterion(output, target)

        return


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.topk(maxk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()


