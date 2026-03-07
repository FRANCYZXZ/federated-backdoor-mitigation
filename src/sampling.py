import random
import numpy as np
from torchvision import datasets, transforms
import torch

random.seed(7)

def distribute_dataset(dataset_name, num_peers, num_classes, dd_type='IID', classes_per_peer=1, samples_per_class=582, alpha=1):
    print(f"--> Loading {dataset_name} dataset...")
    tokenizer = None
    
    if dataset_name == 'MNIST':
        trainset, testset = get_mnist()
    elif dataset_name == 'CIFAR10':
        trainset, testset = get_cifar10()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # Data Distribution Logic
    if dd_type == 'IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=1000000)
    elif dd_type == 'MILD_NON_IID':
        peers_data_dict = sample_dirichlet(trainset, num_peers, alpha=alpha)
    elif dd_type == 'EXTREME_NON_IID':
        peers_data_dict = sample_extreme(trainset, num_peers, num_classes, classes_per_peer, samples_per_class)

    print("--> Dataset successfully distributed!")
    return trainset, testset, peers_data_dict, tokenizer
    
def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_cifar10():
    data_dir = './data'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) # Normalizzazione corretta CIFAR10
    ])
    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return trainset, testset

def sample_dirichlet(dataset, num_users, alpha=1):
    """
    Crea una distribuzione dei dati sui client basata sulla distribuzione di Dirichlet.
    Un alpha alto (es. 1000000) genera una distribuzione IID.
    Un alpha basso (es. 0.5) genera una distribuzione fortemente Non-IID.
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
   
    # Converte i set in liste per compatibilità JSON/YAML
    for i in range(num_users):
        peers_data_dict[i]['labels'] = list(peers_data_dict[i]['labels'])
        
    return peers_data_dict

def sample_extreme(dataset, num_users, num_classes, classes_per_peer, samples_per_class):
    """
    Crea una distribuzione estremamente sbilanciata dove ogni peer
    riceve solo un sottoinsieme specifico di classi.
    """
    n = len(dataset)
    peers_data_dict = {i: {'data': np.array([], dtype=np.int64), 'labels': []} for i in range(num_users)}
    idxs = np.arange(n)
    
    # Gestione compatibilità PyTorch
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