import copy
import numpy as np


def average_weights(w, marks):
    """
    Computes the weighted average of local model weights to produce the global model.
    Used by FedAvg with uniform weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] * (1.0 / sum(marks))
    return w_avg