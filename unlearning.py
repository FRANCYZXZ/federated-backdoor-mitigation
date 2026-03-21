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
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())

def load_reconstructed_image(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    img_pil = Image.open(path).convert('RGB')
    return transform(img_pil).unsqueeze(0).to(device)

def main():
    # 1. Caricamento della configurazione centralizzata
    try:
        with open(REPO_ROOT / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config.yaml' not found. Please run from the project root.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Estrazione dei parametri
    poisoned_ckpt = resolve_repo_path(config['unlearning']['poisoned_checkpoint'])
    sanitized_ckpt = resolve_repo_path(config['unlearning']['sanitized_checkpoint'])
    recon_image = resolve_repo_path(config['unlearning']['recon_image'])
    source_class = config['dataset']['source_class']
    model_name = config['model']['name']
    lr = config['unlearning']['lr']
    epochs = config['unlearning']['epochs']

    print(f"[*] Starting Unlearning Pipeline on {device.upper()}")
    print(f"[*] Poisoned Model: {poisoned_ckpt}")
    print(f"[*] Reconstructed Image: {recon_image}")

    if not os.path.exists(poisoned_ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {poisoned_ckpt}")

    # 2. Inizializzazione modello
    model = setup_model(model_name, num_classes=10, tokenizer=None, embedding_dim=100)
    model.to(device)

    checkpoint = torch.load(poisoned_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))

    # 3. Setup tensori
    img_tensor = load_reconstructed_image(recon_image, device)
    target_tensor = torch.tensor([source_class]).to(device)

    # 4. Fine-Tuning Correttivo (Unlearning) con Batch Mixing
    model.train()
    
    # Blocco parametri BatchNorm per stabilità
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr, momentum=0.9)
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

    for epoch in range(epochs):
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
            
            mixed_data = torch.cat([img_tensor, clean_data], dim=0)
            mixed_targets = torch.cat([target_tensor, clean_targets], dim=0)
        else:
            mixed_data = img_tensor
            mixed_targets = target_tensor

        outputs = model(mixed_data)
        loss = criterion(outputs, mixed_targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:02d}/{epochs} - Loss: {loss.item():.4f}")

    # 5. Salvataggio
    model.eval()
    state = {
        'state_dict': model.state_dict(),
        'unlearning_epochs': epochs,
        'info': 'Model sanitized'
    }
    
    os.makedirs(os.path.dirname(sanitized_ckpt), exist_ok=True)
    
    torch.save(state, sanitized_ckpt)
    
    print(f"[*] Sanitized model saved to {sanitized_ckpt}")

if __name__ == "__main__":
    main()