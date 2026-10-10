"""Machine unlearning methods for a backdoored model."""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from models.models import setup_model

REPO_ROOT = Path(__file__).resolve().parent

def resolve_repo_path(path_str: str) -> str:
    """Resolve a path relative to the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())

def load_reconstructed_image(path: str, device: str) -> torch.Tensor:
    """
    Loads and transforms the reconstructed image into a tensor suitable for training.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    img_pil = Image.open(path).convert('RGB')
    return transform(img_pil).unsqueeze(0).to(device)

def main() -> None:
    """Main execution function for machine unlearning."""
    # 1. Load centralized configuration
    try:
        with open(REPO_ROOT / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config.yaml' not found. Please run from the project root.")
        sys.exit(1)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Extract parameters
    poisoned_ckpt: str = resolve_repo_path(config['unlearning']['poisoned_checkpoint'])
    sanitized_ckpt: str = resolve_repo_path(config['unlearning']['sanitized_checkpoint'])
    recon_image: str = resolve_repo_path(config['unlearning']['recon_image'])
    target_class: int = config['dataset']['target_class']
    model_name: str = config['model']['name']
    unlearning_config = config['unlearning']
    ascent_lr: float = unlearning_config['ascent_lr']
    ascent_steps: int = unlearning_config['ascent_steps']
    ascent_loss_cap: float = unlearning_config['ascent_loss_cap']
    clean_weight: float = unlearning_config['clean_weight']

    print(f"[*] Starting Unlearning Pipeline on {device.upper()}")
    print("[*] Unlearning method: ascent")
    print(f"[*] Poisoned Model: {poisoned_ckpt}")
    print(f"[*] Reconstructed Image: {recon_image}")

    if not os.path.exists(poisoned_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {poisoned_ckpt}")

    # 2. Initialize model
    model = setup_model(model_name, num_classes=10, tokenizer=None, embedding_dim=100)
    model.to(device)

    checkpoint = torch.load(poisoned_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))

    # 3. Setup tensors
    img_tensor = load_reconstructed_image(recon_image, device)
    trigger_target_tensor = torch.tensor([target_class]).to(device)

    # 4. Corrective Fine-Tuning (Unlearning) with Batch Mixing
    model.train()
    
    # Freeze BatchNorm parameters for stability
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ascent_lr,
        momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    # Setup clean data loader to mix with reconstructed image
    transform_clean = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    try:
        clean_dataset = CIFAR10(root=str(REPO_ROOT / 'data'), train=True, download=True, transform=transform_clean)
        clean_loader = DataLoader(clean_dataset, batch_size=31, shuffle=True, drop_last=True)
        clean_iter = iter(clean_loader)
    except Exception as e:
        print(f"Warning: Could not load clean dataset for mixing ({e}). Using only reconstructed image.")
        clean_loader = None

    steps = ascent_steps
    log_interval = max(1, min(5, steps))
    for step in range(steps):
        optimizer.zero_grad()
        
        # Batch Mixing: 1 reconstructed image + 31 clean images
        if clean_loader is not None:
            try:
                clean_data, clean_targets = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                clean_data, clean_targets = next(clean_iter)
            
            clean_data = clean_data.to(device)
            clean_targets = clean_targets.to(device)
            
        trigger_output = model(img_tensor)
        trigger_loss = criterion(trigger_output, trigger_target_tensor)
        if trigger_loss.item() > ascent_loss_cap:
            print(
                f"    Step {step + 1:02d}/{steps} - Early stopping: "
                f"trigger CE {trigger_loss.item():.4f} exceeds cap "
                f"{ascent_loss_cap:.4f}"
            )
            break

        clean_loss = torch.zeros((), device=device)
        if clean_loader is not None:
            clean_output = model(clean_data)
            clean_loss = criterion(clean_output, clean_targets)
        loss = -trigger_loss + clean_weight * clean_loss

        loss.backward()
        optimizer.step()

        if (step + 1) % log_interval == 0 or step == 0 or step + 1 == steps:
            target_probability = torch.softmax(trigger_output.detach(), dim=1)[0, target_class].item()
            print(
                f"    Step {step + 1:02d}/{steps} - "
                f"Trigger CE: {trigger_loss.item():.4f} - "
                f"Clean CE: {clean_loss.item():.4f} - "
                f"Target probability: {target_probability:.4f}"
            )

    # 5. Saving
    model.eval()
    state = {
        'state_dict': model.state_dict(),
        'unlearning_method': 'ascent',
        'unlearning_steps': steps,
        'info': 'Model sanitized'
    }
    sanitized_ckpt_path = Path(sanitized_ckpt)
    sanitized_ckpt = str(
        sanitized_ckpt_path.with_name(
            f"{sanitized_ckpt_path.stem}_ascent{sanitized_ckpt_path.suffix}"
        )
    )
    
    os.makedirs(os.path.dirname(sanitized_ckpt), exist_ok=True)
    
    torch.save(state, sanitized_ckpt)
    
    print(f"[*] Sanitized model saved to {sanitized_ckpt}")

if __name__ == "__main__":
    main()