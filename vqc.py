"""
Implementation of VQC as a pennylane circuit


TODO:
- Finish the basic algorithm
- implement Cross validation loss
"""
import os
import multiprocessing
import pennylane as qml
import time
import jax
import jax.numpy as jnp
import json
from datetime import datetime
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, SPSAOptimizer
from pennylane.templates import AngleEmbedding

from tsp import solve
from data_perm import WineDataset, Dataset, BalancedILPDDataset, ILPDDataset, DiabetesDataset, HeartFailureDataset, Fertility, TransfusionDataset, TeachingAssistantDataset, HayesRothDataset, BanknoteDataset, AcuteInflammationsDataset, ImmunotherapyDataset

import sys

from ansatze import layer, rot_ansatz
from feature_maps import zz_featuremap, z_featuremap, angle_embedding
from loss_functions import cross_entropy_loss, square_loss

class VQCModel(object):
    def __init__(self, dataset, params={}):
        self.dataset = dataset
        self.params = params
        if params:
            self.feature_map = params['feature_map']
            self.ansatz = params['ansatz']
            self.learning_rate = params['learning_rate']
            self.perm_mode = params['perm_mode']
            self.circular = params['circular']
            self.loss = params['loss']
        self.num_layers = params['num_layers'] if params and 'num_layers' in params else 4
        self.num_qubits = params['num_qubits'] if params and 'num_qubits' in params else 10
        self.name = params['name'] if params and 'name' in params else 'default'
        self.useBias = params['useBias'] if params and 'useBias' in params else True
        self.acc = []
        self.c = []
        self.weights = []
        self.dev = qml.device("lightning.qubit", wires = self.num_qubits, shots = 2 ** 10, batch_obs=True)
        self.weights_init = 2.0 * np.pi * np.random.randn(self.num_layers, self.num_qubits*2, 3) if not 'weights_init' in params else params['weights_init']
        self.bias_init = 0.0 if not 'bias_init' in params else params['bias_init']
        self.loaded = False

    def store_params(self):
        params = self.params.copy()
        params['name'] = self.name
        params['feature_map'] = params['feature_map'].__name__
        params['ansatz'] = params['ansatz'].__name__
        params['loss'] = params['loss'].__name__
        params['bias'] = self.bias
        params['loaded'] = self.loaded
        params['weights'] = self.weights.tolist()


        if not os.path.exists('params'):
            os.mkdir('params')
        x = datetime.now()
        s = os.path.dirname(os.path.abspath(__file__)) + r"\params\\" + x.strftime(f'%d-%m-%Y-%H-%M-%S-params-{self.dataset.name}-{self.name}.json')
        f = open(s, 'w')
        json.dump(params, f, indent=4)
        f.close()
    
    def store_results(self):
        result = {}
        if len(self.acc) > 0:
            result['init'] = [self.acc[0], self.c[0]]
        for i in range(1, len(self.acc)):
            result[i] = [self.acc[i], self.c[i]]
        if not os.path.exists('results'):
            os.mkdir('results')
        x = datetime.now()
        s = os.path.dirname(os.path.abspath(__file__)) + r"\results\\" + x.strftime(f'%d-%m-%Y-%H-%M-%S-results-{self.dataset.name}-{self.name}.json')
        f = open(s, 'w')
        json.dump(result, f, indent=4)
        f.close()
        
    def load(self, params):
        res = isinstance(params, str)
        if res:
            with open(params) as json_file:
                params = json.load(json_file)
        if 'weights' in params:
            params['weights_init'] = params['weights']
            self.weights_init = params['weights']
        if 'bias' in params:
            params['bias_init'] = params['bias']
            self.bias_init = params['bias']

        match params['feature_map']:
            case 'z_featuremap':
                params['feature_map'] = z_featuremap
            case 'zz_featuremap':
                params['feature_map'] = zz_featuremap
            case 'angle_embedding':
                params['feature_map'] = angle_embedding

        match params['ansatz']:
            case 'layer':
                params['ansatz'] = layer
            case 'rot_ansatz':
                params['ansatz'] = rot_ansatz

        match params['loss']:
            case 'square_loss':
                params['loss'] = square_loss
            case 'cross_entropy_loss':
                params['loss'] = cross_entropy_loss
        
        self.params = params
        self.feature_map = params['feature_map']
        self.ansatz = params['ansatz']
        self.learning_rate = params['learning_rate']
        self.num_layers = params['num_layers']
        self.perm_mode = params['perm_mode']
        self.circular = params['circular']
        self.loss = params['loss']
        self.num_qubits = params['num_qubits']
        self.name = params['name']
        self.useBias = params['useBias']
        self.dev = qml.device("lightning.qubit", wires = self.num_qubits, shots = 2 ** 10, batch_obs=True)
        self.loaded = True


    def train(self, epochs=251, f_iter=1):
        if not os.path.exists('logs'):
            os.mkdir('logs')
        x = datetime.now()
        s = os.path.dirname(os.path.abspath(__file__)) + r"\logs\\" + x.strftime(f'%d-%m-%Y-%H-%M-%S-log-{self.dataset.name}-{self.name}.txt')
        f = open(s, 'w')

        train_data, train_labels = self.dataset.get_separated_data()
        test_data, test_labels = self.dataset.get_separated_test_data()

        var_mat = np.abs(np.corrcoef(self.dataset.data[:,:-1], rowvar=False))

        max_perm, dist = solve(var_mat, maximize = True, linear = False)
        min_perm, dist = solve(var_mat, linear = False)
        rand_perm = np.arange(self.num_qubits)
        np.random.shuffle(rand_perm)

        match self.perm_mode:
            case 'max':
                perm_train_data = train_data[:, max_perm]
            case 'min':
                perm_train_data = train_data[:, min_perm]
            case 'random':
                perm_train_data = train_data[:, rand_perm]
            case _:
                perm_train_data = train_data[:, min_perm]

        opt = AdamOptimizer(self.learning_rate)
        self.weights = np.copy(self.weights_init)
        self.bias = self.bias_init if self.useBias else None

        pred = [self.variational_classifier(self.weights, self.bias, x, self.feature_map, self.ansatz, self.circular, f_iter) for x in perm_train_data]
        predictions = self.get_sign(pred)

        self.acc.append(self.accuracy(train_labels, predictions))
        self.c.append(self.loss(train_labels, pred))

        print("Initial values     - acc: {0} - c: {1}, {2}, {3}".format(self.acc[0], self.c[0], self.dataset.name, self.name))
        f.write(f'Initial values     - acc: {self.acc[0]} - c: {self.c[0]}, {self.name}\n')
        for i in range(epochs):
            results = opt.step(self.cost, self.weights, self.bias, self.circular, f_iter, self.loss, train_data, train_labels)
            self.weights = results[0]
            self.bias = results[1] if self.useBias else None

            pred = [self.variational_classifier(self.weights, self.bias, x, self.feature_map, self.ansatz, self.circular, f_iter) for x in perm_train_data]
            predictions = self.get_sign(pred)
            print(np.unique(predictions, return_counts=True))

            self.acc.append(self.accuracy(train_labels, predictions))
            self.c.append(self.loss(train_labels, pred))

            if i % 50 == 0:
                self.store_params()
                self.store_results()

            print("Epoch: {0} - norm - acc: {1} - c: {2}, {3}".format(i+1, self.acc[-1], self.c[-1], self.dataset.name))
            f.write(f"Epoch: {i+1} - norm - acc: {self.acc[-1]} - c: {self.c[-1]}\n")
        self.store_params()
        self.store_results()
        f.close()

    def cost(self, weights, bias, circular, f_iter, loss, train_data, labels):
        predictions = [self.variational_classifier(weights = weights, bias = bias, x = x, feature_map=self.feature_map, ansatz=self.ansatz, circular=circular, f_iter=f_iter) for x in train_data]
        return loss(labels, predictions)

    def get_sign(self, v):
        n_vals = np.sign(v)

        for i in range(len(v)):
            if n_vals[i] == 0.0:
                n_vals[i] = 1.0

        return n_vals

    def accuracy(self, labels, predictions):
        acc = 0

        for l, p in zip(labels, predictions):
            if abs(l-p) < 1e-5:
                acc += 1

        acc = acc / len(labels)

        return acc

    def circuit_wrapper(self, weights, x, circular, f_iter, feature_map, ansatz):
        @qml.qnode(self.dev, interface="autograd", diff_method='parameter-shift')
        def circuit(weights, x, circular, f_iter, feature_map, ansatz):

            for w in weights:
                feature_map(x=x, f_iter=f_iter, num_qubits=self.num_qubits)
                ansatz(W=w, circular=circular, num_qubits=self.num_qubits)

            return qml.expval(qml.PauliZ(0))
        return circuit(weights, x, circular, f_iter, feature_map, ansatz)

    def variational_classifier(self, weights, bias, x, feature_map, ansatz, circular = True, f_iter = 1):
        bias_val = 0.0
        if bias is not None:
            bias_val = bias

        return self.circuit_wrapper(weights = weights, x=x, circular=circular, f_iter=f_iter, feature_map=feature_map, ansatz=ansatz) + bias_val

if __name__ == "__main__":
    # datasets = [WineDataset(), BalancedILPDDataset(), ILPDDataset(), DiabetesDataset(), HeartFailureDataset(), Fertility(), TransfusionDataset(), TeachingAssistantDataset(), HayesRothDataset(), BanknoteDataset(), AcuteInflammationsDataset(), ImmunotherapyDataset()]
    datasets = [HayesRothDataset()]


    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    jobs = []
    for d in datasets:
        options = {'feature_map': z_featuremap,
               'ansatz': layer,
               'learning_rate': 0.02,
               'num_layers': 5,
               'perm_mode': 'random',
               'loss': cross_entropy_loss,
               'circular': True,
               'num_qubits': d.num_qubits,
               'name': 'with_bias'
               }
        vqc = VQCModel(d, options)
        p = multiprocessing.Process(target=vqc.train)
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()


