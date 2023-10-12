"""
Implementation of VQC as a pennylane circuit


TODO:
- Finish the basic algorithm
- implement Cross validation loss
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer
from pennylane.templates import AngleEmbedding

import matplotlib.pyplot as plt

from tsp import solve
from data_perms import Dataset, ILPDDataset

import sys
import os

num_qubits = 10
dev = qml.device("default.qubit", wires = num_qubits, shots = 1000)

def layer(W, circular):
    """
    EfficientSU2 ansatz, using circular
    """
    for i in range(num_qubits):
        qml.RY(W[i, 0], wires=i)
        qml.RZ(W[i, 1], wires=i)

    for i in range(num_qubits):
        if i + 1 != num_qubits or circular:
            qml.CNOT(wires=[i, (i + 1) % num_qubits])

def state_preparation(x, f_iter):
    """
    ZFeatureMap
    """
    for j in range(f_iter):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.U1(2.0*x[i], wires=i)

@qml.qnode(dev, interface="autograd")
def circuit(weights, x, circular, f_iter):
    state_preparation(x=x, f_iter=f_iter)

    for w in weights:
        layer(W=w, circular=circular)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, x, circular = True, f_iter = 1):
    return circuit(weights = weights, x=x, circular=circular, f_iter=f_iter)

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

def accuracy(labels, predictions):
    acc = 0

    for l, p in zip(labels, predictions):
        if abs(l-p) < 1e-5:
            acc += 1

    acc = acc / len(labels)

    return acc

def cost(weights, circular, f_iter, loss, train_data, labels):
    predictions = [variational_classifier(weights = weights, x = x, circular=circular, f_iter=f_iter) for x in train_data]
    return loss(labels, predictions)

if __name__ == "__main__":
    dataset = ILPDDataset(neg_labels = True)
    num_layers = 3

    train_data, train_labels = dataset.get_separated_data()
    test_data, test_labels = dataset.get_separated_test_data()

    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 2)
    lr = 0.1

    f_iter = 3
    circular = True
    loss = cross_entropy_loss

    opt = AdamOptimizer(lr)
    weights = np.copy(weights_init)

    epochs = 20

    acc = []
    c = []
    pred = [variational_classifier(weights, x, circular, f_iter) for x in train_data]
    predictions = np.sign(pred)

    acc.append(accuracy(train_labels, predictions))
    c.append(loss(train_labels, pred))

    print("Initial values     - acc: {0} - c: {1}".format(acc[0], c[0]))

    for i in range(epochs):
        weights = opt.step(cost, weights, circular, f_iter, loss, train_data, train_labels)[0]

        pred = [variational_classifier(weights, x, circular, f_iter) for x in train_data]
        predictions = np.sign(pred)

        acc.append(accuracy(train_labels, predictions))
        c.append(loss(train_labels, pred))

        print("Epoch: {0} - norm - acc: {1} - c: {2}".format(i+1, acc[-1], c[-1]))
        os.system("notify-send 'Epoch done'")

    figs, axes = plt.subplots(2)
    axes[0].set_title("Objective function values")
    axes[1].set_title("Accuracy values")

    axes[0].plot(range(len(c)), c, color = "green")

    axes[1].plot(range(len(acc)), acc, color = "green")

    plt.show()
