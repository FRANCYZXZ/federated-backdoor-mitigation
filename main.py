"""
Main entry point for the Federated Learning Security framework.
Handles configuration parsing, environment setup, and triggers the FL simulation.
"""

from __future__ import print_function
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import random
import yaml
import sys
from typing import Dict

from engine.experiment_federated import run_exp

REPO_ROOT = Path(__file__).resolve().parent

try:
    with open(REPO_ROOT / "config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: 'config.yaml' file not found in the project root.")
    sys.exit(1)

SEED: int = config['training']['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION: nn.Module = nn.CrossEntropyLoss()

LABELS_DICT: Dict[str, int] = {
    'Plane': 0, 'Car': 1, 'Bird': 2, 'Cat': 3, 'Deer': 4,
    'Dog': 5, 'Frog': 6, 'Horse': 7, 'Ship': 8, 'Truck': 9
}

def main() -> None:
    """
    Main execution loop.
    Iterates through the attacker ratios defined in the configuration and 
    runs the federated learning experiment for each setting.
    """
    recon_mode: bool = config.get('execution', {}).get('reconstruction_only', False)

    for atr in config['attack']['attackers_ratio']:
        print("="*60)
        print(f" STARTING EXPERIMENT | Attacker Ratio: {atr} | Rule: {config['federated']['rule']}")
        print(f" Mode -> Reconstruction Only: {recon_mode}")
        print("="*60)
        
        run_exp(
            dataset_name=config['dataset']['name'], 
            model_name=config['model']['name'], 
            dd_type=config['federated']['dd_type'], 
            num_peers=config['federated']['num_peers'], 
            frac_peers=config['federated']['frac_peers'], 
            seed=SEED, 
            test_batch_size=config['training']['test_batch_size'],
            criterion=CRITERION, 
            global_rounds=config['training']['global_rounds'], 
            local_epochs=config['training']['local_epochs'], 
            local_bs=config['training']['local_bs'], 
            local_lr=config['training']['local_lr'], 
            local_momentum=config['training']['local_momentum'], 
            labels_dict=LABELS_DICT, 
            device=DEVICE,
            attackers_ratio=atr, 
            attack_type=config['attack']['type'], 
            malicious_behavior_rate=config['attack']['malicious_behavior_rate'], 
            rule=config['federated']['rule'],
            source_class=config['dataset']['source_class'], 
            target_class=config['dataset']['target_class'],
            class_per_peer=config['federated']['class_per_peer'], 
            samples_per_class=config['federated']['samples_per_class'], 
            rate_unbalance=config['federated']['rate_unbalance'], 
            alpha=config['federated']['alpha'], 
            reconstruction_only=recon_mode
        )

if __name__ == "__main__":
    main()