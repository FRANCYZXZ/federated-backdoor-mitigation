"""
Module to initialize and execute the federated learning environment.
"""

import numpy as np
from typing import Dict, Any, Optional

from engine.environment_federated import FL


def run_exp(dataset_name: str, model_name: str, dd_type: str,
            num_peers: int, frac_peers: float, seed: int, test_batch_size: int, criterion: Any, global_rounds: int, 
            local_epochs: int, local_bs: int, local_lr: float, local_momentum: float, labels_dict: Dict[str, int], device: Any, 
            attackers_ratio: float, attack_type: str, malicious_behavior_rate: float, rule: str, 
            class_per_peer: int, samples_per_class: int, rate_unbalance: float, alpha: float, source_class: int, target_class: int, resume: bool,
            reconstruction_only: bool = False) -> None:
    """
    Sets up the Federated Learning environment and runs the experiment.
    """
    print('\n--> Starting experiment...')
    
    # Initialize FL Environment
    flEnv = FL(dataset_name=dataset_name, model_name=model_name, dd_type=dd_type, num_peers=num_peers, 
               frac_peers=frac_peers, seed=seed, test_batch_size=test_batch_size, criterion=criterion, global_rounds=global_rounds, 
               local_epochs=local_epochs, local_bs=local_bs, local_lr=local_lr, local_momentum=local_momentum, 
               labels_dict=labels_dict, device=device, attackers_ratio=attackers_ratio,
               class_per_peer=class_per_peer, samples_per_class=samples_per_class, 
               rate_unbalance=rate_unbalance, alpha=alpha, source_class=source_class)
    
    print('Dataset:', dataset_name)
    print('Data distribution:', dd_type)
    print('Aggregation rule:', rule)
    print('Attack Type:', attack_type)
    print('Attackers Ratio:', np.round(attackers_ratio*100, 2), '%')
    print('Malicious Behavior Rate:', malicious_behavior_rate*100, '%')
    
    if reconstruction_only:
        print('MODE: Reconstruction / Attack Simulation ONLY (No Training)')
    
    # Execute
    flEnv.run_experiment(attack_type=attack_type, malicious_behavior_rate=malicious_behavior_rate, 
                         source_class=source_class, target_class=target_class, 
                         rule=rule, resume=resume, 
                         reconstruction_only=reconstruction_only) 
                    
    print('\n--> End of Experiment.')