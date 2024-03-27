import global_vars

from torchvision.transforms import Normalize, Compose, RandomHorizontalFlip, ToTensor, Resize
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
import pandas as pd


class TinyImageNetVal(Dataset):
    def __init__(self, root_dir, annotation_file, class_to_idx, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_csv(os.path.join(root_dir, annotation_file), sep='\t', header=None, usecols=[0, 1])
        self.class_to_idx = class_to_idx  # Mapping from class IDs to integer indices

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # Convert the class ID to an integer index
        label = self.class_to_idx[self.annotations.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)


def getLoaders(datasetName, gen, input_size=None):
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

    elif datasetName == 'imagenet-tiny':
        # Data loading code
        img_size = 224
        if global_vars.args.dummy:
            print("=> Dummy data is used!")
            train_dataset = datasets.FakeData(128, (3, img_size, img_size), 200, transforms.ToTensor())
            val_dataset = datasets.FakeData(64, (3, img_size, img_size), 200, transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
                num_workers=global_vars.args.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
                num_workers=global_vars.args.workers, pin_memory=True)

        else:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),  # Tiny ImageNet images are 64x64
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            val_transform = transforms.Compose([
                transforms.Resize(256),  # Tiny ImageNet images are 64x64
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            traindir = os.path.join(global_vars.args.data_path, 'train')
            valdir = os.path.join(global_vars.args.data_path, 'val')

            train_dataset = datasets.ImageFolder(root=traindir, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=global_vars.args.batch_size,
                                      shuffle=True, num_workers=global_vars.args.workers)
            class_to_idx = train_dataset.class_to_idx
            val_dataset = TinyImageNetVal(root_dir=valdir,
                                          annotation_file='val_annotations.txt',
                                          class_to_idx=class_to_idx,
                                          transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=global_vars.args.batch_size,
                                    shuffle=False, num_workers=global_vars.args.workers)

        n_classes = 200
        shape = (3, img_size, img_size)
        return train_loader, val_loader, n_classes, shape
    elif datasetName == 'imagenet-1k':
        # Data loading code
        if global_vars.args.dummy:
            print("=> Dummy data is used!")
            train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
            val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
        else:
            traindir = os.path.join(global_vars.args.data_path, 'train')
            valdir = os.path.join(global_vars.args.data_path, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

        train_sampler = None
        val_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
            num_workers=global_vars.args.workers, pin_memory=True, sampler=train_sampler, generator=gen)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=global_vars.args.batch_size, shuffle=False,
            num_workers=global_vars.args.workers, pin_memory=True, sampler=val_sampler)
        n_classes = 1000
        shape = (3, 224, 224)
        return train_loader, val_loader, n_classes, shape
    else:
        raise Exception("dataset isn't supported.")
