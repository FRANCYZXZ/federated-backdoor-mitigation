import copy
import torch
import numpy as np
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_pca(data):
    """
    Applica la Principal Component Analysis per ridurre la dimensionalità
    delle similarità del coseno tra i gradienti.
    """
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def average_weights(w, marks):
    """
    Esegue la media pesata dei pesi locali per ottenere il modello globale.
    Utilizzata sia da FedAvg (con pesi uniformi) che da FL-Defender (con pesi basati sul trust).
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1.0 / sum(marks))
    return w_avg


class FLDefender:
    """
    Meccanismo di difesa reattivo per il Federated Learning.
    Calcola l'affidabilità (trust score) dei client misurando la distanza
    tra l'aggiornamento del singolo client e il centroide degli aggiornamenti.
    """
    def __init__(self, num_peers):
        self.grad_history = None
        self.fgrad_history = None
        self.num_peers = num_peers

    def score(self, global_model, local_models, peers_types, selected_peers, epoch, tau=1.5):
        global_model = list(global_model.parameters())
        last_g = global_model[-2].cpu().data.numpy()
        m = len(local_models)
        f_grads = [None for i in range(m)]
        
        # Estrazione degli ultimi layer (gradienti fittizi)
        for i in range(m):
            grad = (last_g - list(local_models[i].parameters())[-2].cpu().data.numpy())
            f_grads[i] = grad.reshape(-1)
    
        # Calcolo della similarità del coseno tra i gradienti dei peer
        cs = smp.cosine_similarity(f_grads) - np.eye(m)
        cs = get_pca(cs)
        
        # Calcolo del centroide e assegnazione dello score
        centroid = np.median(cs, axis=0)
        scores = smp.cosine_similarity([centroid], cs)[0]

        return scores