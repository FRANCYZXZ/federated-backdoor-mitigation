import numpy as np
import torch
from torch.utils import data
from torchvision import datasets, transforms

class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class=None, target_class=None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.dataset)

def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)

def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        current_indices = list(dict_users[i])
        current_labels = [dataset[idx][1] for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        current_indices = list(dict_users[i])
        current_labels = [int(dataset[idx][1]) for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

def distribute_dataset(dataset_name, num_peers, num_classes, dd_type, class_per_peer, samples_per_class, alpha):
    tokenizer = None
    user_groups_train = {}
    train_dataset, test_dataset = None, None

    if dataset_name == 'CIFAR10':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=trans_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=trans_test)

        if dd_type == 'IID':
            user_groups_train = cifar_iid(train_dataset, num_peers)
        else:
            user_groups_train = cifar_iid(train_dataset, num_peers)

    elif dataset_name == 'MNIST':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=trans_mnist)

        user_groups_train = mnist_iid(train_dataset, num_peers)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return train_dataset, test_dataset, user_groups_train, tokenizer