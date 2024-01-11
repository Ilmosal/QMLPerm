"""
Implementation of VQC as a pennylane circuit


TODO:
- Finish the basic algorithm
- implement Cross validation loss
"""
import pennylane as qml
import jax
import jax.numpy as jnp
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, SPSAOptimizer
from pennylane.templates import AngleEmbedding

import matplotlib.pyplot as plt

from tsp import solve
from data_perm import Dataset, BalancedILPDDataset, ILPDDataset, WineDataset

import sys
import os

num_qubits = 10
dev = qml.device("default.qubit", wires = num_qubits, shots = 1000)

def phi(x, y):
    """
    Nonlinear function phi
    """
    return (np.pi - x) * (np.pi - y)

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

def rot_ansatz(W, circular):
    """
    Ansatz with full three direction rotations
    """
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

    for i in range(num_qubits):
        if i + 1 != num_qubits or circular:
            qml.CRot(W[num_qubits+i, 0], W[num_qubits+i, 1], W[num_qubits+i, 2], wires=[i, (i + 1) % num_qubits])


def zz_featuremap(x, f_iter):
    """
    ZZFeatureMap
    """
    for j in range(f_iter):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.U1(2.0*x[i], wires=i)

        for x in range(num_qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.U1(2.0*phi(x[i], x[i+1]), wires=i+1)
            qml.CNOT(wires=[i, i+1])

def z_featuremap(x, f_iter):
    """
    ZFeatureMap
    """
    for j in range(f_iter):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.U1(np.pi*x[i], wires=i)

def angle_embedding(x):
    for i in range(num_qubits):
        qml.RX(x[i]*np.pi, wires=i)

@qml.qnode(dev, interface="autograd")
def circuit(weights, x, circular, f_iter):
    z_featuremap(x=x, f_iter=f_iter)
    #angle_embedding(x)

    for w in weights:
        #layer(W=w, circular=circular)
        rot_ansatz(W=w, circular=circular)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x, circular = True, f_iter = 1):
    bias_val = 0.0
    if bias is not None:
        bias_val = bias

    return circuit(weights = weights, x=x, circular=circular, f_iter=f_iter) + bias_val

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

def cost(weights, bias, circular, f_iter, loss, train_data, labels):
    predictions = [variational_classifier(weights = weights, bias = bias, x = x, circular=circular, f_iter=f_iter) for x in train_data]
    return loss(labels, predictions)

def get_sign(v):
    n_vals = np.sign(v)

    for i in range(len(v)):
        if n_vals[i] == 0.0:
            n_vals[i] = 1.0

    return n_vals

if __name__ == "__main__":
    dataset = BalancedILPDDataset()
    num_layers = 4

    train_data, train_labels = dataset.get_separated_data()
    test_data, test_labels = dataset.get_separated_test_data()

    var_mat = np.abs(np.corrcoef(dataset.data[:,:-1], rowvar=False))

    max_perm, dist = solve(var_mat, maximize = True, linear = False)
    min_perm, dist = solve(var_mat, linear = False)

    min_train_data = train_data[:, min_perm]

    #weights_init = 2.0 * np.pi * np.random.randn(num_layers, num_qubits, 2)
    weights_init = 2.0 * np.pi * np.random.randn(num_layers, num_qubits*2, 3)
    bias_init = 0.0

    lr = 0.01

    f_iter = 1
    circular = True
    loss = cross_entropy_loss

    opt = AdamOptimizer(lr)
    weights = np.copy(weights_init)
    bias = bias_init

    epochs = 250

    acc = []
    c = []
    pred = [variational_classifier(weights, bias, x, circular, f_iter) for x in min_train_data]
    predictions = get_sign(pred)

    acc.append(accuracy(train_labels, predictions))
    c.append(loss(train_labels, pred))

    print("Initial values     - acc: {0} - c: {1}".format(acc[0], c[0]))

    for i in range(epochs):
        results = opt.step(cost, weights, bias, circular, f_iter, loss, train_data, train_labels)
        weights = results[0]
        bias = results[1]

        pred = [variational_classifier(weights, bias, x, circular, f_iter) for x in min_train_data]
        predictions = get_sign(pred)
        print(np.unique(predictions, return_counts=True))

        acc.append(accuracy(train_labels, predictions))
        c.append(loss(train_labels, pred))

        print("Epoch: {0} - norm - acc: {1} - c: {2}".format(i+1, acc[-1], c[-1]))

    figs, axes = plt.subplots(2)
    axes[0].set_title("Objective function values")
    axes[1].set_title("Accuracy values")

    axes[0].plot(range(len(c)), c, color = "green")

    axes[1].plot(range(len(acc)), acc, color = "green")

    plt.show()
