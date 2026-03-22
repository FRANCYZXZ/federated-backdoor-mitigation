# Federated Backdoor Mitigation: Backdoor Attacks, Gradient Inversion and Machine Unlearning

This framework provides an environment for simulating and analyzing security threats in Federated Learning (FL). The project explores the interaction between model integrity attacks (Backdoor) and data privacy breaches (Gradient Inversion), implementing a reactive defense strategy based on Machine Unlearning.

## Technical Overview

The system implements the following operational phases:

1.  **Backdoor Attack Simulation:** Malicious clients inject poisoned patterns (pixel-pattern poisoning) to manipulate the global model's behavior towards a target class.
2.  **Federated Training (FedAvg):** Standard `fedavg` aggregation is used, intentionally leaving the model vulnerable to demonstrate the attack's effectiveness.
3.  **Gradient Inversion Attack:** Reconstruction of clients' private training images from shared gradient updates sent to the server.
4.  **Machine Unlearning:** A model sanitization procedure that utilizes reconstructed images, mixed with clean data batches, to eliminate the backdoor trigger via corrective fine-tuning.

## Architecture and Configuration

The entire framework is centrally managed by the `config.yaml` file. This approach separates execution logic from experimental parameters.

### Key Configuration Variables (`config.yaml`)
- `dataset`: Control the source/target attack classes (e.g. source 1, target 0).
- `federated`: 
  - `rule`: Aggregation rule (`fedavg`).
  - `num_peers` / `frac_peers`: Number of total clients and fraction selected per round.
  - `alpha`: Controls data heterogeneity (Dirichlet distribution).
- `training`: Standard FL hyperparameters (`global_rounds`, `local_bs`, `local_lr`, etc.).
- `attack`:
  - `attackers_ratio`: Array of fractions of malicious clients.
  - `malicious_behavior_rate`: Probability a selected attacker will actually poison their update in a given round.
- `unlearning`: Paths to the poisoned model, the reconstructed trigger image, and the target output for the sanitized model.
- `execution`: 
  - `reconstruction_only`: Skip FL training entirely and only run gradient inversion on the target checkpoint.

---

## Installation & Execution

The framework can be run either locally via standard Python or via Docker (recommended for consistency). 

> **Hardware Note:** Gradient Inversion (Phase 2) is highly computationally expensive. A CUDA-enabled GPU is strongly recommended to run these phases within a reasonable timeframe.

### Option A: Running with Docker (Recommended)
You can build and execute the framework using Docker Compose, which automatically provisions the necessary PyTorch environment.

1.  **Standard Training / Attack Simulation** (Runs `main.py`):
    ```bash
    docker compose up --build
    ```

2.  **Run Machine Unlearning** (Overriding the default command):
    ```bash
    docker compose run --rm fl-security python3 unlearning.py
    ```

3.  **Run Prediction / Verification**:
    ```bash
    docker compose run --rm fl-security python3 predict.py
    ```

### Option B: Local Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/FRANCYZXZ/federated-backdoor-mitigation.git
    cd federated-backdoor-mitigation
    ```
2.  Install the required dependencies (A virtual environment is recommended):
    ```bash
    pip install -r requirements.txt
    ```

## Workflow

The project is divided into independent modules coordinated by the configuration file.

### Phase 1: Federated Training
Train the global model under attack using the `fedavg` aggregation rule.
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
The final stage evaluates the Attack Success Rate (ASR) and global model accuracy against the reconstructed triggers.
```bash
python predict.py
```

## Directory Structure

To maintain a clean environment, the framework automatically organizes outputs as follows (these paths are preserved as volumes when using Docker):

```text
model_checkpoints/
├── checkpoints/       # Intermediate training states (.t7)
├── results/           # Final trained models
└── sanitized_model/   # Models post-Machine Unlearning
reconstructed_images/  # Original and reconstructed samples from Phase 2
```

## References
- **Inverting Gradients**: Privacy breach simulation based on the algorithm by Jonas Geiping (*NeurIPS 2020*).