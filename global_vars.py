import logging

from torch import device, Generator, save, load
from torch.cuda import is_available
from argparse import ArgumentParser
from copy import deepcopy
from argparse import Namespace
from copy import copy
import json
import shutil
import os

parser = ArgumentParser(description='PyTorch GN\AGN Training')
parser.add_argument('--dataset', default="CIFAR100", type=str,
                    help='the dataset which we learn on (default: CIFAR100)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--method', default="GN", type=str,
                    help='choose the method for running (default: GroupNorm)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('-rbs', '--reclustring_bs', default=512, type=int,
                    metavar='N',
                    help='mini-batch sifze for channels reclustring (default: 2048)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--eps', default=1e-12, type=float,
                    help='normalization epsilon')
parser.add_argument('--group_by_size', default=False, type=bool,
                    help='if to normalize by group whith the same size')
parser.add_argument('--group_norm', default=32, type=int,
                    help='number of groups, in the GN/SGN/RGN layers (default: 32)')
parser.add_argument('--group_norm_size', default=16, type=int,
                    help='number of channels per group, in the norm layers (default: 16)')
parser.add_argument('--norm_shuffle', default=10, type=int, metavar='N',
                    help='the number that our normalization layer is'
                         + ' reclustring its channels groups (default: every 10 epochs)')
parser.add_argument('--max_norm_shuffle', default=1000, type=int, metavar='N',
                    help='the number which our normalization layer is not'
                         + ' reclustring its channels groups anymore. (default: 1000')
parser.add_argument('--riar', default=1, type=int,
                    help='how meny times to reclustring in a row (default: 1)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 1000)')
parser.add_argument('-e', '--evaluate', default=False, type=bool,
                    help='evaluate model on validation set')
parser.add_argument('--load', default=None, type=str,
                    help='the path to model that you want to initial the model weights (default: None)')
parser.add_argument('--save_weights', default=None, type=str,
                    help='the path you want to save initial the model weights (default: None)')
parser.add_argument('--saveing_path', default=None, type=str,
                    help='the path you want to save the model weights after each training loop (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--scheduler_name', default="default", type=str,
                    metavar='PATH',
                    help='for future scheduler selection (default: "default")')
parser.add_argument('--base_scheduler_name', default="default", type=str,
                    metavar='PATH',
                    help='relevant when scheduler_name=myCustomScheduler')
parser.add_argument('--seed', default=0, type=int,
                    help='fix all random behivers to random state (default: 0)')
parser.add_argument('--GN_in_bt', default=False, action="store_true",
                    help='Even if RGN/SGN is set, run GN in the bottle-neck')
parser.add_argument('--use_k_means', default=False, action="store_true",
                    help='use k-means algo when norm method is SGN')
parser.add_argument('--shuf_each_batch', default=False, action="store_true",
                    help='re-cluster the channels each batch')
parser.add_argument('--classes_to_train', nargs='+', default=[],
                    help='two names of classes (like bear/wolf')
parser.add_argument('--plot_std', default=False, action="store_true",
                    help='plot std for first batch')
parser.add_argument('--save_shuff_idxs', default=False, action="store_true",
                    help='Save shuffle indexes of the full epoch and apply them on the rest of the epoch')
parser.add_argument('--config', type=str, default='',
                    help='path to json file of training configuration')
parser.add_argument('--SGN_version', default=1, type=int,
                    help='SGN metric version.'
                         '1: grouping as far channels, (mean/var)*(mean+var), range in group: numGroups'
                         '2: grouping as far channels, (mean/var)*(mean+var), range in group: numberOfChannels/groupSize'
                         '3: grouping as close channels, (mean/var)*(mean+var)'
                         '4: grouping as close channels, KMeans'
                         '5: groups only in the same batch[i], (mean/var)*(mean+var)'
                         '6: grouping as close channels, (std)'
                         '7: grouping using diffusion maps'
                         '8: grouping using harmonic mean')
parser.add_argument('--RGN_version', default=1, type=int,
                    help='RGN metric version.')
parser.add_argument('--model_version', default=1, type=int,
                    help='Model version'
                         '1: Original ResNet50'
                         '2: 1 + one dropout before last fc')
parser.add_argument('--code_version', default=1, type=int, help='Code version')
parser.add_argument('--use_wandb', default=False, action="store_true",
                    help='use weights&biases')
parser.add_argument('--epoch_start_cluster', default=0,
                    help='epoch number to start cluster from')
parser.add_argument('--cluster_last_batch', default=False, action="store_true",
                    help='when getting to epoch of clustering, cluster at'
                         'last epoch. default=cluster in the first epoch')
parser.add_argument('--dropout_prop', default=0.2, type=float,
                    help='dropout probability')



def apply_config(args: Namespace, config_path: str):
    """Overwrite the values in an arguments object by values of namesake
    keys in a JSON config file.

    :param args: The arguments object
    :param config_path: the path to a config JSON file.
    """
    config_path = copy(config_path)
    if config_path:
        # Opening JSON file
        f = open(config_path)
        config_overwrite = json.load(f)
        for k, v in config_overwrite.items():
            if k.startswith('_'):
                continue
            setattr(args, k, v)


def save_config_in_saving_path(args):
    if args.saveing_path and os.path.exists(
            args.saveing_path) and args.config != "" and os.path.exists(
            args.config):
        version = ""
        if args.method == "SGN":
            version = f"_V{args.SGN_version}"
        elif args.method == "RGN":
            version = f"_V{args.RGN_version}"

        new_name = f"config_{args.method}{version}.json"
        shutil.copyfile(args.config, os.path.join(args.saveing_path, new_name))
    else:
        print("Note: didn't save config file!")


# set global_vars free variables
args = parser.parse_args()

apply_config(args, args.config)

save_config_in_saving_path(args)

is_agn = args.method == "RGN" or args.method == "SGN"
device_name = 'cuda' if is_available() else 'cpu'
device = device(device_name)

recluster = True
train_mode = True
before_shuffle = True
best_state_dict = None
zero_state_dict = None
best_prec1 = 0

generator = Generator()
generator.manual_seed(args.seed)
epoch_num = 0

# initialize the results arrays
train_prcition1, train_prcition5 = [], []
test_prcition1, test_prcition5 = [], []
train_losses, test_losses = [], []


def save_results(train_loss, train_prc1, train_prc5, test_loss, test_prc1,
                 test_prc5):
    train_losses.append(train_loss)
    train_prcition1.append(train_prc1)
    train_prcition5.append(train_prc5)
    test_losses.append(test_loss)
    test_prcition1.append(test_prc1)
    test_prcition5.append(test_prc5)


def save_initial_weights():
    if args.save_weights is not None:
        print("saving the initial weights in path: ", args.save_weights)
        save({'state_dict': zero_state_dict}, args.save_weights)


def save_checkpoint(model, optimizer, epoch):
    # save best prec@1
    save_best_prec1(model)

    # save checkpoint
    save({'epoch': epoch,
          'arch': args.arch,
          'model': model,
          'Zero_state_dict': zero_state_dict,
          'Best_state_dict': best_state_dict,
          'best_prec1': best_prec1,
          'optimizer': optimizer.state_dict(),
          'train_losses': train_losses,
          'train_prcition1': train_prcition1,
          'train_prcition5': train_prcition5,
          'test_losses': test_losses,
          'test_prcition1': test_prcition1,
          'test_prcition5': test_prcition5,
          # 'generator': generator,
          },
         args.saveing_path
         )


def save_best_prec1(model):
    global best_prec1
    if test_prcition1[-1] > best_prec1:
        best_state_dict = deepcopy(model.state_dict())
        best_prec1 = test_prcition1[-1]


def printParameters():
    print('architecture = ', args.arch)
    print('method = ', args.method)
    print('workers size = ', args.workers)
    print('epochs = ', args.epochs)
    print('start epoch = ', args.start_epoch)
    print('seed = ', args.seed)
    print('batch size = ', args.batch_size)
    print('lr = ', args.lr)
    print('scheduler name = ', args.scheduler_name)
    print('momentum = ', args.momentum)
    print('eps = ', args.eps)
    print('group norm by size = ', args.group_by_size)
    print('group norm number of groups = ', args.group_norm)
    print('group norm size = ', args.group_norm_size)
    print('norm shuffle = ', args.norm_shuffle)
    print('max norm shuffle = ', args.max_norm_shuffle)
    print('weight decay = ', args.weight_decay)
    print('print freq = ', args.print_freq)
    print('evaluate = ', args.evaluate)
    print('number of clustring in a row = ', args.riar)
    print('load = ', args.load)
    print('save weights = ', args.save_weights)
    print('saveing_path = ', args.saveing_path)
    print('resume = ', args.resume)
    print('GN in bottle-neck = ', args.GN_in_bt)


def init_saveing_path():
    if (args.saveing_path == None):
        args.saveing_path = "//content"
    args.saveing_path = args.saveing_path + '//{}'.format(args.arch)
    args.saveing_path = args.saveing_path + '_{}'.format(args.method)
    if args.method == 'SGN':
        args.saveing_path = args.saveing_path + 'V{}'.format(args.SGN_version)
    elif args.method == 'RGN':
        args.saveing_path = args.saveing_path + 'V{}'.format(args.RGN_version)
    args.saveing_path = args.saveing_path + '{}'.format(args.norm_shuffle)
    args.saveing_path = args.saveing_path + '_{}b'.format(args.batch_size)
    args.saveing_path = args.saveing_path + '_{}lr'.format(args.lr)
    args.saveing_path = args.saveing_path + '_{}m'.format(args.momentum)
    if not args.group_by_size:
        args.saveing_path = args.saveing_path + '_{}groups'.format(
            args.group_norm)
    else:
        args.saveing_path = args.saveing_path + '_{}groupsize'.format(
            args.group_norm_size)
    args.saveing_path = args.saveing_path + '_{}reclustringinarow'.format(
        args.riar)
    args.saveing_path = args.saveing_path + '_{}eps'.format(args.eps)
    args.saveing_path = args.saveing_path + '_{}seed'.format(args.seed)
    args.saveing_path = args.saveing_path + '_{}maxreclustring'.format(
        args.max_norm_shuffle)
    args.saveing_path = args.saveing_path + '_{}'.format(args.scheduler_name)
    args.saveing_path = args.saveing_path + '.tar'


def generate_wandb_name():
    wanda_test_name = f"{args.method}"

    if is_agn:
        if args.method == 'RGN':
            wanda_test_name += f'_V{args.RGN_version}'
        else:
            wanda_test_name += f'_V{args.SGN_version}'

        if args.epoch_start_cluster != 0:
            wanda_test_name += f'_epoch-start-cluster-{args.epoch_start_cluster}'

        if args.cluster_last_batch:
            wanda_test_name += f'_cluster-last_batch'

        wanda_test_name += f'_shuff-every-ep-{args.norm_shuffle}'

    wanda_test_name += f'_bs-{args.batch_size}'

    if args.group_by_size:
        wanda_test_name += f'_gs-{args.group_norm_size}'
    else:
        wanda_test_name += f'_num-of-groups-{args.group_norm}'
    
    if args.model_version == 2:
        wanda_test_name += f'_du-{args.dropout_prop}'

    if args.load:
        wanda_test_name += f'_{os.path.splitext(os.path.basename(args.load))[0]}'

    return wanda_test_name


def load_checkpoint(model, optimizer):
    checkpoint = load(args.resume, map_location=device)
    train_losses, test_losses = checkpoint['train_losses'], checkpoint[
        'test_losses']
    train_prcition1, train_prcition5 = checkpoint['train_prcition1'], \
                                       checkpoint['train_prcition5']
    test_prcition1, test_prcition5 = checkpoint['test_prcition1'], checkpoint[
        'test_prcition5']
    zero_state_dict = checkpoint['Zero_state_dict']
    best_state_dict = checkpoint['Best_state_dict']
    best_prec1 = max(test_prcition1)
    args.start_epoch = checkpoint['epoch'] + 1
    # optimizer.load_state_dict(checkpoint['optimizer'])
    model = checkpoint['model']
    # with torch.no_grad():
    #     for param1, param2 in zip(model.state_dict(), checkpoint_state_dict):
    #         model.state_dict()[param1].data.fill_(0).add_(checkpoint_state_dict[param2].data)
    train_prcition1, train_prcition5 = checkpoint['train_prcition1'], \
                                       checkpoint['train_prcition5']
    test_prcition1, test_prcition5 = checkpoint['test_prcition1'], checkpoint[
        'test_prcition5']
    train_losses, test_losses = checkpoint['train_losses'], checkpoint[
        'test_losses']
    return model, optimizer


def groupsBySize(numofchannels):
    return (numofchannels / args.group_norm_size)
