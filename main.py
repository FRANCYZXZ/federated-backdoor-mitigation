from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import random
import yaml
import sys

from engine.experiment_federated import run_exp

try:
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Errore: File 'config.yaml' non trovato. Assicurati che sia nella root del progetto.")
    sys.exit(1)

SEED = config['training']['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss()

LABELS_DICT = {
    'Plane':0, 'Car':1, 'Bird':2, 'Cat':3, 'Deer':4,
    'Dog':5, 'Frog':6, 'Horse':7, 'Ship':8, 'Truck':9
}

if __name__ == "__main__":
    
    resume_mode = config.get('execution', {}).get('resume', False)
    recon_mode = config.get('execution', {}).get('reconstruction_only', False)

    for atr in config['attack']['attackers_ratio']:
        print("="*60)
        print(f" AVVIO ESPERIMENTO | Attacker Ratio: {atr} | Rule: {config['federated']['rule']}")
        print(f" Modalità -> Resume: {resume_mode} | Reconstruction Only: {recon_mode}")
        print("="*60)
        
        run_exp(
            dataset_name = config['dataset']['name'], 
            model_name = config['model']['name'], 
            dd_type = config['federated']['dd_type'], 
            num_peers = config['federated']['num_peers'], 
            frac_peers = config['federated']['frac_peers'], 
            seed = SEED, 
            test_batch_size = config['training']['test_batch_size'],
            criterion = CRITERION, 
            global_rounds = config['training']['global_rounds'], 
            local_epochs = config['training']['local_epochs'], 
            local_bs = config['training']['local_bs'], 
            local_lr = config['training']['local_lr'], 
            local_momentum = config['training']['local_momentum'], 
            labels_dict = LABELS_DICT, 
            device = DEVICE,
            attackers_ratio = atr, 
            attack_type = config['attack']['type'], 
            malicious_behavior_rate = config['attack']['malicious_behavior_rate'], 
            rule = config['federated']['rule'],
            source_class = config['dataset']['source_class'], 
            target_class = config['dataset']['target_class'],
            class_per_peer = config['federated']['class_per_peer'], 
            samples_per_class = config['federated']['samples_per_class'], 
            rate_unbalance = config['federated']['rate_unbalance'], 
            alpha = config['federated']['alpha'], 
            resume = resume_mode,
            reconstruction_only = recon_mode
        )