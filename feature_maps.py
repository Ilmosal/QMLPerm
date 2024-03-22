import pennylane as qml
from pennylane import numpy as np

def phi(x, y):
    """
    Nonlinear function phi
    """
    return (np.pi - x) * (np.pi - y)


def zz_featuremap(x, f_iter, num_qubits):
    """
    ZZFeatureMap
    """

    for j in range(f_iter):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.U1(2.0*x[i], wires=i)

        for i in range(num_qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.U1(2.0*phi(x[i], x[i+1]), wires=i+1)
            qml.CNOT(wires=[i, i+1])

def z_featuremap(x, f_iter, num_qubits):
    """
    ZFeatureMap
    """
    for j in range(f_iter):
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
            qml.U1(np.pi*x[i], wires=i)

def angle_embedding(x, f_iter, num_qubits):
    for i in range(num_qubits):
        qml.RX(x[i]*np.pi, wires=i)