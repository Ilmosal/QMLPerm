import pennylane as qml

def layer(W, circular, num_qubits):
    """
    EfficientSU2 ansatz, using circular
    """
    for i in range(num_qubits):
        qml.RY(W[i, 0], wires=i)
        qml.RZ(W[i, 1], wires=i)

    for i in range(num_qubits):
        if i + 1 != num_qubits or circular:
            qml.CNOT(wires=[i, (i + 1) % num_qubits])

def rot_ansatz(W, circular, num_qubits):
    """
    Ansatz with full three direction rotations
    """
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

    for i in range(num_qubits):
        if i + 1 != num_qubits or circular:
            qml.CRot(W[num_qubits+i, 0], W[num_qubits+i, 1], W[num_qubits+i, 2], wires=[i, (i + 1) % num_qubits])