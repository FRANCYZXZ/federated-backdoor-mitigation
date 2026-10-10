"""
Inference script to evaluate the reconstructed images against the sanitized models.
"""

import sys
import argparse
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from models.models import setup_model
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

REPO_ROOT = Path(__file__).resolve().parent

def resolve_repo_path(path_str: str) -> str:
    """Resolve a path relative to the repository root."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())

def main() -> None:
    """Main execution function for inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Override predict.checkpoint_path")
    parser.add_argument("--image", help="Override predict.image_path")
    parser.add_argument(
        "--dataset-metrics",
        action="store_true",
        help="Also evaluate clean accuracy and trigger ASR on CIFAR-10",
    )
    args = parser.parse_args()
    try:
        with open(REPO_ROOT / "config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: 'config.yaml' file not found.")
        sys.exit(1)

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    recon_image_path: str = resolve_repo_path(args.image or config['predict']['image_path'])
    checkpoint_path: str = resolve_repo_path(args.checkpoint or config['predict']['checkpoint_path'])
    target_class_idx: int = config['dataset']['target_class']
    source_class_idx: int = config['dataset']['source_class']
    model_name: str = config['model']['name']

    labels_map = {
        0: 'Plane', 1: 'Car', 2: 'Bird', 3: 'Cat', 4: 'Deer', 
        5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
    }

    print(f"Analyzing image: {recon_image_path}")
    print(f"Using model: {checkpoint_path}")

    model = setup_model(model_name, num_classes=10, tokenizer=None, embedding_dim=100)
    model.to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    try:
        img_pil = Image.open(recon_image_path).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image not found at {recon_image_path}")
        sys.exit(1)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    pred_idx: int = predicted_idx.item()
    pred_label: str = labels_map.get(pred_idx, str(pred_idx))
    conf_pct: float = confidence.item() * 100

    print("\n" + "="*40)
    print("INFERENCE RESULT")
    print("="*40)
    print(f"Actual Class (Source): {labels_map.get(source_class_idx, 'Unknown')}")
    print(f"Target Class (Target): {labels_map.get(target_class_idx, 'Unknown')}")
    print("-" * 40)
    print(f"PREDICTION: {pred_label.upper()}")
    print(f"Confidence: {conf_pct:.2f}%")
    print("-" * 40)

    if pred_idx == target_class_idx:
        print("ATTACK SUCCESS: The model recognized the backdoor.")
    elif pred_idx == source_class_idx:
        print("ATTACK FAILED: The model recognized the original object.")
    else:
        print("UNCERTAIN RESULT: The model predicted a third class.")

    if args.dataset_metrics:
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        eval_dataset = CIFAR10(
            root=str(REPO_ROOT / 'data'),
            train=False,
            download=True,
            transform=eval_transform,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
        correct = 0
        total = 0
        backdoor_correct = 0
        backdoor_total = 0
        backdoor_pattern = torch.tensor(
            [[[2.5141] * 3] * 3, [[2.5968] * 3] * 3, [[2.7537] * 3] * 3],
            dtype=torch.float32,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            for data, labels in eval_loader:
                data, labels = data.to(device), labels.to(device)
                predictions = model(data).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()

                source_mask = labels == source_class_idx
                if source_mask.any():
                    triggered = data[source_mask].clone()
                    triggered[:, :, -3:, -3:] = backdoor_pattern
                    target_predictions = model(triggered).argmax(dim=1)
                    backdoor_correct += (target_predictions == target_class_idx).sum().item()
                    backdoor_total += target_predictions.numel()

        clean_accuracy = 100.0 * correct / total
        asr = 100.0 * backdoor_correct / backdoor_total if backdoor_total else 0.0
        print(f"CLEAN ACCURACY: {clean_accuracy:.2f}%")
        print(f"ASR: {asr:.2f}%")

if __name__ == "__main__":
    main()