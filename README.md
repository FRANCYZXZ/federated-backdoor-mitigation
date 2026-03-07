# Federated Learning Security: Backdoor Attacks, Gradient Inversion and Machine Unlearning

This framework provides an environment for simulating and analyzing security threats in Federated Learning (FL). The project explores the interaction between model integrity attacks (Backdoor) and data privacy breaches (Gradient Inversion), implementing a reactive defense strategy based on Machine Unlearning.

## Technical Overview

The system implements the following operational phases:

1.  **Backdoor Attack Simulation:** Malicious clients inject poisoned patterns (pixel-pattern poisoning) to manipulate the global model's behavior towards a target class.
2.  **Robust Aggregation:** Comparison between standard `fedavg` (vulnerable) and the `fl_defender` protocol, designed to filter malicious gradients during training.
3.  **Gradient Inversion Attack:** Reconstruction of clients' private training images from shared gradient updates sent to the server.
4.  **Machine Unlearning:** A model sanitization procedure that utilizes reconstructed images to eliminate the backdoor trigger via corrective fine-tuning.



## Architecture and Configuration

The entire framework is centrally managed by a `config.yaml` file. This approach separates execution logic from experimental parameters, allowing the modification of datasets, models, aggregation rules, and attack settings without altering the source code.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/FRANCYZXZ/Federated-Learning-Security-Backdoor-Attacks-Gradient-Inversion-Unlearning.git](https://github.com/FRANCYZXZ/Federated-Learning-Security-Backdoor-Attacks-Gradient-Inversion-Unlearning.git)
cd Federated-Learning-Security-Backdoor-Attacks-Gradient-Inversion-Unlearning
pip install -r requirements.txt
```

## Workflow

The project is divided into independent modules coordinated by the configuration file.

### Phase 1: Federated Training
Train the global model under attack.
- Set `rule: "fedavg"` in the config to observe a successful attack.
- Set `rule: "fl_defender"` to test preventive defense.

```bash
python main.py
```

### Phase 2: Gradient Inversion Reconstruction
This phase simulates a privacy breach where a curious server attempts to recover private training data from shared gradients. It targets the malicious client's updates to reconstruct the specific poisoned image used for the attack.

To execute, set `reconstruction_only: true` in `config.yaml` and run:

```bash
python main.py
```

### Phase 3: Machine Unlearning

This module performs a reactive defense. Once the poisoned image is reconstructed, the global model undergoes a targeted fine-tuning process (unlearning). This procedure forces the model to associate the malicious trigger with its original correct class, effectively neutralizing the backdoor.

```bash
python unlearning.py
```

### Phase 4: Verification and Inference

The final stage evaluates the Attack Success Rate (ASR) and global model accuracy. It verifies the effectiveness of the chosen defense (either the preventive FL-Defender or the reactive Unlearning) by testing the model against the reconstructed triggers.

```bash
python predict.py
```

## Directory Structure

To maintain a clean environment, the framework automatically organizes outputs as follows:

```text
model_checkpoints/
├── checkpoints/       # Intermediate training states (.t7)
├── results/           # Final trained models (vulnerable or robust)
└── sanitized_model/   # Models post-Machine Unlearning
reconstructed_images/  # Original and reconstructed samples from Phase 2
```
## References

  - **FL-Defender**: Preventive defense mechanism based on research by Najeeb Jebreel (FL-Defender: Combating Targeted Attacks in Federated Learning).

  - **Inverting Gradients**: Privacy breach simulation based on the algorithm by Jonas Geiping (Inverting Gradients – How Easy Is It to Break Privacy in Federated Learning?, NeurIPS 2020).