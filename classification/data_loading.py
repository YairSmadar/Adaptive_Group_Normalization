import global_vars

from torchvision.transforms import Normalize, Compose, RandomHorizontalFlip, ToTensor, Resize
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch

"""
0: apple 1: aquarium_fish 2: baby 3: bear 4: beaver

5: bed 6: bee 7: beetle 8: bicycle 9: bottle 10: bowl

11: boy 12: bridge 13: bus 14: butterfly 15: camel

16: can 17: castle 18: caterpillar 19: cattle 20: chair

21: chimpanzee 22: clock 23: cloud 24: cockroach 25: couch

26: cra 27: crocodile 28: cup 29: dinosaur 30: dolphin

31: elephant 32: flatfish 33: forest 34: fox 35: girl

36: hamster 37: house 38: kangaroo 39: keyboard 40: lamp 41: lawn_mower

42: leopard 43: lion 44: lizard 45: lobster 46: man 47: maple_tree 

48: motorcycle 49: mountain 50: mouse 51: mushroom 52: oak_tree

53: orange 54: orchid 55: otter 56: palm_tree 57: pear

58: pickup_truck 59: pine_tree 60: plain 61: plate 62: poppy

63: porcupine 64: possum 65: rabbit 66: raccoon 67: ray

68: road 69: rocket 70: rose 71: sea 72: seal 73: shark

74: shrew 75: skunk 76: skyscraper 77: snail 78: snake

79: spider 80: squirrel 81: streetcar 82: sunflower 83: sweet_pepper

84: table 85: tank 86: telephone 87: television 88: tiger 89: tractor

90: train 91: trout 92: tulip 93: turtle 94: wardrobe 95: whale

96: willow_tree 97: wolf 98: woman 99: worm
"""


def getLoaders(datasetName, gen, input_size=None):
    reclustringLoader = None
    if datasetName == 'cifar100':
        CIFAR100_normalize = Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        # Check if input_size is provided and add Resize transformation accordingly
        if input_size is not None:
            train_transform = Compose([Resize(input_size), RandomHorizontalFlip(), ToTensor(), CIFAR100_normalize])
            test_transform = Compose([Resize(input_size), ToTensor(), CIFAR100_normalize])
        else:
            train_transform = Compose([RandomHorizontalFlip(), ToTensor(), CIFAR100_normalize])
            test_transform = Compose([ToTensor(), CIFAR100_normalize])
        trainset = CIFAR100(root='./dataset/train', train=True, download=True, transform=train_transform)
        testset = CIFAR100(root='./dataset/test', train=False, download=True, transform=test_transform)

        # # histogram of all means of each class
        # means_dic = {}
        # for i in range(len(trainset.class_to_idx)):
        #     classes_in = np.where(np.array(trainset.targets) == i)
        #     data = trainset.data[classes_in, :, :, :].squeeze()
        #     # n, w, h, c = data.shape
        #     # data = data.reshape((n, w*h, c))
        #     # channel_dist = torch.cat([torch.from_numpy(data[i, :, :]) for i in range(n)], dim=1)
        #     # mean = channel_dist.cpu().detach().numpy().mean(axis=1)
        #     mean = data.mean()
        #     means_dic[i] = mean
        # sort_means_dict = dict(sorted(means_dic.items(), key=lambda item: item[1]))

        if len(global_vars.args.classes_to_train) > 0 and 'all' not in global_vars.args.classes_to_train:

            train_list = []
            test_list = []
            for _class in global_vars.args.classes_to_train:

                class_idx = trainset.class_to_idx[_class]

                # handle train
                class_data = np.where(np.array(trainset.targets) == class_idx)
                train_list.append(class_data)

                # handle test
                class_data = np.where(np.array(testset.targets) == class_idx)
                test_list.append(class_data)

            index_train = np.concatenate(train_list, axis=1).squeeze()
            index_train = np.sort(index_train)
            trainset.data = trainset.data[index_train, :, :, :]
            trainset.targets = [trainset.targets[i] for i in index_train]

            index_test = np.concatenate(test_list, axis=1).squeeze()
            index_test = np.sort(index_test)
            testset.data = testset.data[index_test, :, :, :]
            testset.targets = [testset.targets[i] for i in index_test]

        train_loader = DataLoader(trainset, batch_size=global_vars.args.batch_size, shuffle=True,
                                  num_workers=global_vars.args.workers, pin_memory=True, generator=gen)
        val_loader = DataLoader(testset, batch_size=global_vars.args.batch_size, shuffle=False,
                                num_workers=global_vars.args.workers, pin_memory=True)
        H, W, C = train_loader.dataset.data.shape[1:]
        return train_loader, val_loader, len(train_loader.dataset.classes), (C, H, W)

    elif datasetName == 'imagenet':
        # Data loading code
        img_size = 224
        if global_vars.args.dummy:
            print("=> Dummy data is used!")
            train_dataset = datasets.FakeData(128, (3, img_size, img_size), 200, transforms.ToTensor())
            val_dataset = datasets.FakeData(64, (3, img_size, img_size), 200, transforms.ToTensor())
        else:
            traindir = os.path.join(global_vars.args.data_path, 'train')
            valdir = os.path.join(global_vars.args.data_path, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    normalize,
                ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
            num_workers=global_vars.args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
            num_workers=global_vars.args.workers, pin_memory=True)

        n_classes = 200
        shape = (3, img_size, img_size)
        return train_loader, val_loader, n_classes, shape
    else:
        raise Exception("dataset isn't supported.")
