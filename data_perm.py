"""
Python script for determining how data permutations affect performance of quantum machine learinig

TODO
- load dataset
- dataset arrangement iterator
- apply learning algorithm
    - Quantum kernel or variational circuit?
"""
from python_tsp.heuristics import solve_tsp_simulated_annealing
from sklearn.model_selection import train_test_split
from scipy.io import arff

import numpy as np


#from qiskit import BasicAer
#from qiskit.algorithms.optimizers import COBYLA, SPSA
#from qiskit.circuit.library import EfficientSU2, ZFeatureMap
#from qiskit_machine_learning.algorithms.classifiers import VQC

import pennylane as qml

from tsp import solve, total_dist

class Dataset(object):
    """
    Dataset object to contain the dataset
    """

    def __init__(self, dataset_path, test_size = 0.2, neg_labels = True):
        # load data
        self.raw_data = np.loadtxt(dataset_path, delimiter=",", dtype=float)


        # normalize data
        self.data = np.copy(self.raw_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        np.random.shuffle(self.data)

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

    def get_separated_test_data(self):
        if self.test.size == 0:
            return np.array([]), np.array([])

        test_data = self.test[:,:-1]
        test_labels = []
        for i in range(len(self.test)):
            test_labels.append(int(self.test[i][-1]))

        return test_data, test_labels

    def get_separated_data(self):
        train_data = self.train[:,:-1]
        train_labels = []

        for i in range(len(self.train)):
            train_labels.append(int(self.train[i][-1]))

        return train_data, np.array(train_labels)

    def get_max_ordered_data(self, linear):
        var_mat = np.abs(np.corrcoef(self.data[:,:-1], rowvar=False))
        max_perm, dist = solve(var_mat, maximize=True, linear=linear)

        train_data, train_labels = self.get_separated_data()

        return train_data[:, max_perm], train_labels, max_perm, dist

    def get_min_ordered_data(self, linear):
        var_mat = np.abs(np.corrcoef(self.data[:,:-1], rowvar=False))
        min_perm, dist = solve(var_mat, maximize=False, linear=linear)

        train_data, train_labels = self.get_separated_data()

        return train_data[:, min_perm], train_labels, min_perm, dist

class WineDataset(Dataset):
    """
    Container for the Wine Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/wine.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Wine'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 1.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class BanknoteDataset(Dataset):
    """
    Container for the Banknote authentication Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.5):
        tmp_data = np.loadtxt("./datasets/banknote_authentication.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Banknote'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 1.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class ImmunotherapyDataset(Dataset):
    """
    Container for the Banknote authentication Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/immunotherapy.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Immunotherapy'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 1.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class AcuteInflammationsDataset(Dataset):
    """
    Container for the Acute inflammations Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/acute_inflammations.csv", delimiter=',', dtype=str)
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Inflammations'

        for i in range(len(tmp_data)):
            tmp_data[i, 0] = float(tmp_data[i, 0])
            for j in range(1, 8):
                tmp_data[i,j] = 1.0 if tmp_data[i,j] == 'no' else 0.0
        tmp_data = [[eval(y) for y in x] for x in tmp_data]
        self.raw_data = tmp_data

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class TransfusionDataset(Dataset):
    """
    Container for the Transfusion Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.5):
        tmp_data = np.loadtxt("./datasets/transfusion.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Transfusion'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 1.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class HayesRothDataset(Dataset):
    """
    Container for the Hayes-Roth Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/hayes-roth.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Hayes-Roth'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 2.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class TeachingAssistantDataset(Dataset):
    """
    Container for the Hayes-Roth Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/teaching_assistant.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'TeachingAssistant'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 2.0 if tmp_data[i,-1] == 2.0 else 1.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class DiabetesDataset(Dataset):
    """
    Container for the Wine Dataset
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/diabetes_data_upload.csv", delimiter=',', dtype=str)
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Diabetes'

        for i in range(len(tmp_data)):
            tmp_data[i, 0] = int(tmp_data[i, 0])
            for j in range(1, 17):
                tmp_data[i,j] = 1.0 if tmp_data[i,j] == 'Negative' or tmp_data[i,j] == 'No' or tmp_data[i,j] == 'Female' else 0.0

        tmp_data = [[eval(y) for y in x] for x in tmp_data]
        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class HeartFailureDataset(Dataset):
    """
    Container for the heart failure
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/heart_failure_clinical_records_dataset.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'HeartFailure'

        for i in range(len(tmp_data)):
            tmp_data[i,-1] = 1.0 if tmp_data[i,-1] == 1.0 else 0.0

        self.raw_data = tmp_data

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class Fertility(Dataset):
    """
    Container for the heart failure
    """
    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = np.loadtxt("./datasets/fertility.csv", delimiter=',')
        self.num_qubits = len(tmp_data[0]) - 1
        self.name = 'Fertility'

        # Normalize
        self.data = np.copy(tmp_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        np.random.shuffle(self.data)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class ILPDDataset(Dataset):
    """
    Container for the ILDP dataset
    """

    def __init__(self, neg_labels = True, test_size = 0.0):
        tmp_data = arff.loadarff("./datasets/ilpd.arff")[0]
        tr_data = []
        self.num_qubits = 10
        self.name = 'ILPD'
        for d in tmp_data:
            entry = [
                d[0],
                0.0 if d[1] == b'Male' else 1.0,
                d[2],
                d[3],
                d[4],
                d[5],
                d[6],
                d[7],
                d[8],
                d[9],
                0.0 if d[10] == b'1' else 1.0
            ]

            tr_data.append(entry)

        self.raw_data = np.array(tr_data)

        # normalize data
        self.data = np.copy(self.raw_data)
        self.data -= np.min(self.data, axis=0)
        self.data /= np.ptp(self.data, axis=0)

        if neg_labels:
            for i in range(len(self.data)):
                self.data[i, -1] = self.data[i, -1] * 2 - 1

        np.random.shuffle(self.data)

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)

class BalancedILPDDataset(Dataset):
    """
    Version of ILPD dataset with balanced label amounts for testing
    """
    def __init__(self, test_size = 0.0):
        tmp = ILPDDataset(neg_labels=True)
        self.num_qubits = 10
        self.raw_data = tmp.raw_data
        self.name = 'ILPDBalanced'
        lab_1 = []
        lab_2 = []

        for i in range(len(tmp.data)):
            if tmp.data[i,-1] == 1.0:
                lab_1.append(tmp.data[i])
            else:
                lab_2.append(tmp.data[i])

        num_vals = len(lab_1)

        lab_2 = lab_2[:num_vals]

        tmp_dat = lab_1 + lab_2

        self.data = np.array(tmp_dat)
        np.random.shuffle(self.data)

        # Split to training and validation set
        if test_size == 0.0:
            self.train = self.data
            self.test = np.array([])
        else:
            self.train, self.test = train_test_split(self.data, test_size=test_size)


