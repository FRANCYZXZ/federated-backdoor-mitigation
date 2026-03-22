import numpy as np
import torch
import random
from torch.utils import data
from torchvision import datasets, transforms
from typing import Tuple, Dict, Any, Optional

class CustomDataset(data.Dataset):
    """
    Custom dataset wrapper to handle label switching for targeted attacks.
    """
    def __init__(self, dataset: data.Dataset, indices: np.ndarray, source_class: Optional[int] = None, target_class: Optional[int] = None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self) -> int:
        return len(self.indices)

class PoisonedDataset(data.Dataset):
    """
    Dataset wrapper for poisoned data, applying label switching globally.
    """
    def __init__(self, dataset: data.Dataset, source_class: Optional[int] = None, target_class: Optional[int] = None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)

def combine_datasets(list_of_datasets: list) -> data.Dataset:
    """Concatenate multiple datasets."""
    return data.ConcatDataset(list_of_datasets)

def cifar_iid(dataset: data.Dataset, num_users: int) -> Dict[int, Dict[str, Any]]:
    """Sample IID client data from CIFAR dataset."""
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        current_indices = list(dict_users[i])
        current_labels = [dataset[idx][1] for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

def mnist_iid(dataset: data.Dataset, num_users: int) -> Dict[int, Dict[str, Any]]:
    """Sample IID client data from MNIST dataset."""
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        current_indices = list(dict_users[i])
        current_labels = [int(dataset[idx][1]) for idx in current_indices]
        dict_users[i] = {'data': np.array(current_indices), 'labels': current_labels}
        
    return dict_users

def sample_dirichlet(dataset: data.Dataset, num_users: int, alpha: float = 1.0) -> Dict[int, Dict[str, Any]]:
    """
    Creates a data distribution based on Dirichlet distribution.
    A high alpha (e.g. 1000000) generates an IID distribution.
    A low alpha (e.g. 0.5) generates a highly Non-IID distribution.
    """
    classes = {}
    for idx, x in enumerate(dataset):
        _, label = x
        if type(label) == torch.Tensor:
            label = label.item()
        if label in classes:
            classes[label].append(idx)
        else:
            classes[label] = [idx]
            
    num_classes = len(classes.keys())
    peers_data_dict = {i: {'data': np.array([], dtype=np.int64), 'labels': set()} for i in range(num_users)}

    for n in range(num_classes):
        random.shuffle(classes[n])
        class_size = len(classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(np.array(num_users * [alpha]))
        for user in range(num_users):
            num_imgs = int(round(sampled_probabilities[user]))
            sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
            peers_data_dict[user]['data'] = np.concatenate((peers_data_dict[user]['data'], np.array(sampled_list)), axis=0)
            if num_imgs > 0:
                peers_data_dict[user]['labels'].add(n)

            classes[n] = classes[n][min(len(classes[n]), num_imgs):]
   
    # Convert sets to lists for compatibility
    for i in range(num_users):
        peers_data_dict[i]['labels'] = list(peers_data_dict[i]['labels'])
        
    return peers_data_dict

def sample_extreme(dataset: data.Dataset, num_users: int, num_classes: int, classes_per_peer: int, samples_per_class: int) -> Dict[int, Dict[str, Any]]:
    """
    Creates an extremely unbalanced distribution where each peer 
    receives only a specific subset of classes.
    """
    n = len(dataset)
    peers_data_dict = {i: {'data': np.array([], dtype=np.int64), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([y for _, y in dataset])

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]

    label_indices = {l: [] for l in range(num_classes)}
    for l in label_indices:
        label_idxs = np.where(labels == l)
        label_indices[l] = list(idxs[label_idxs])
    
    available_labels = [i for i in range(num_classes)]

    for i in range(num_users):
        user_labels = np.random.choice(available_labels, classes_per_peer, replace=False)
        for l in user_labels:
            peers_data_dict[i]['labels'].append(l)
            lab_idxs = label_indices[l][:samples_per_class]
            label_indices[l] = list(set(label_indices[l]) - set(lab_idxs))
            
            if len(label_indices[l]) < samples_per_class and l in available_labels:
                available_labels.remove(l)
                
            peers_data_dict[i]['data'] = np.concatenate((peers_data_dict[i]['data'], lab_idxs), axis=0)
    
    return peers_data_dict

def distribute_dataset(dataset_name: str, num_peers: int, num_classes: int, dd_type: str, class_per_peer: int, samples_per_class: int, alpha: float) -> Tuple[data.Dataset, data.Dataset, Dict[int, Dict[str, Any]], Any]:
    """
    Downloads and distributes the requested dataset among peers based on the distribution type.
    """
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

    elif dataset_name == 'MNIST':
        trans_mnist = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=trans_mnist)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=trans_mnist)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    # Apply data partitioning strategy
    if dd_type == 'IID':
        if dataset_name == 'CIFAR10':
            user_groups_train = cifar_iid(train_dataset, num_peers)
        else:
            user_groups_train = mnist_iid(train_dataset, num_peers)
    elif dd_type == 'MILD_NON_IID':
        user_groups_train = sample_dirichlet(train_dataset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        user_groups_train = sample_extreme(train_dataset, num_peers, num_classes, class_per_peer, samples_per_class)
    else:
        # Fallback to dirichlet which handles alpha=1000000 as IID
        user_groups_train = sample_dirichlet(train_dataset, num_peers, alpha=alpha)

    return train_dataset, test_dataset, user_groups_train, tokenizer