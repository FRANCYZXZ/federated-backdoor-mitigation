import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv

class CNNMNIST(nn.Module):
    """
    Semplice rete convoluzionale per il dataset MNIST.
    """
    def __init__(self):
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def setup_model(model_architecture, num_classes=10, tokenizer=None, embedding_dim=None):
    """
    Inizializza e restituisce il modello richiesto.
    (I parametri tokenizer e embedding_dim sono mantenuti solo per 
    compatibilità di firma con vecchie chiamate, ma non vengono utilizzati).
    """
    available_models = {
        "CNNMNIST": CNNMNIST,
        "ResNet18": tv.models.resnet18,
        "VGG16": tv.models.vgg16,
        "DN121": tv.models.densenet121,
    }

    print(f'--> Creazione del modello {model_architecture}...')

    if model_architecture not in available_models:
        print("Errore: Architettura del modello non specificata o non disponibile.")
        raise ValueError(model_architecture)

    model = available_models[model_architecture]()

    # Adatta l'ultimo layer del modello pre-addestrato al numero di classi del nostro dataset
    if "ResNet18" in model_architecture:
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes)
        
    elif "VGG16" in model_architecture:
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, num_classes)
        
    elif "DN121" in model_architecture:
        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, num_classes)

    print('--> Modello creato con successo!')
    return model