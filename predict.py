import sys
import yaml
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from models.models import setup_model

def main():
    try:
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print("Errore: File 'config.yaml' non trovato.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    recon_image_path = config['predict']['image_path']
    checkpoint_path = config['predict']['checkpoint_path']
    target_class_idx = config['dataset']['target_class']
    source_class_idx = config['dataset']['source_class']
    model_name = config['model']['name']

    labels_map = {
        0: 'Plane', 1: 'Car', 2: 'Bird', 3: 'Cat', 4: 'Deer', 
        5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'
    }

    print(f"Analisi dell'immagine: {recon_image_path}")
    print(f"Modello in uso: {checkpoint_path}")

    model = setup_model(model_name, num_classes=10, tokenizer=None, embedding_dim=100)
    model.to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print(f"Errore: Checkpoint non trovato in {checkpoint_path}")
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
        print(f"Errore: Immagine non trovata in {recon_image_path}")
        sys.exit(1)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
    pred_idx = predicted_idx.item()
    pred_label = labels_map.get(pred_idx, str(pred_idx))
    conf_pct = confidence.item() * 100

    print("\n" + "="*40)
    print("RISULTATO INFERENZA")
    print("="*40)
    print(f"Classe Reale (Source): {labels_map.get(source_class_idx, 'Sconosciuta')}")
    print(f"Classe Obiettivo (Target): {labels_map.get(target_class_idx, 'Sconosciuta')}")
    print("-" * 40)
    print(f"PREDIZIONE: {pred_label.upper()}")
    print(f"Confidenza: {conf_pct:.2f}%")
    print("-" * 40)

    if pred_idx == target_class_idx:
        print("SUCCESSO ATTACCO: Il modello ha riconosciuto la backdoor.")
    elif pred_idx == source_class_idx:
        print("ATTACCO FALLITO: Il modello ha riconosciuto l'oggetto originale.")
    else:
        print("RISULTATO INCERTO: Il modello ha predetto una terza classe.")

if __name__ == "__main__":
    main()