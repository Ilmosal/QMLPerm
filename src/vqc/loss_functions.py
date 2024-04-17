from pennylane import numpy as np

def square_loss(labels, predictions):
    loss = 0

    for l, p in zip(labels, predictions):
        loss += (l - p)**2

    return loss / len(labels)

def n2b(x):
    return (x+1) / 2

def cross_entropy_loss(labels, predictions):
    loss = 0

    for l, p in zip(labels, predictions):
        loss += n2b(l) * np.log2(n2b(p)) + (1 - n2b(l)) * np.log2(1 - n2b(p))

    return -loss / len(labels)