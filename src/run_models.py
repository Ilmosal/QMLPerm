"""
Module for running models using data reuploading classifiers to test the effect of data-permutation to QML
"""
import itertools
import json
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from data.bars_and_stripes import generate_bars_and_stripes
from data.hidden_manifold import generate_hidden_manifold_model
from data.hyperplanes import generate_hyperplanes_parity
from data.linearly_separable import generate_linearly_separable
from data.mnist import generate_mnist
from data.two_curves import generate_two_curves

from vqc.tsp import solve
from data_reuploading import DataReuploadingClassifier#, IQPDataReuploading, DataReuploadingRX
from iqp_variational import IQPVariationalClassifier
from iqp_kernel import IQPKernelClassifier
from finBench import create_finbench_cd2, create_finbench_ld1, create_finbench_cf1, create_finbench_cc3
from noise_data import generate_noisy_dataset, noise_data_perms
from alt_kernel import AltKernelClassifier
from model_utils import accuracy

tsp_max_iter = 2000

# List of params for the project
dataset_list = [
        {"dataset-seed": 42, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": 100, "dimension": 10, "margin": 0.2, "gen_noise": False, "n_train":240, "n_test":60, "permutation":None},
        {"dataset-seed": 1234, "dataset": "bars-and-stripes", "order-seed": 1234, "models-trained": 100,"height": 4, "width": 5, "noise-std":0.5, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation":None},
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": 100, "dimension": 20, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
        {"dataset-seed": 1234, "dataset": "two-curves", "order-seed": 1234, "models-trained": 100, "n_features": 20, "degree":5, "offset":0.1, "noise":0.01, "gen_noise": False, "n_train":300, "n_test":300, "permutation":None},
        {"dataset-seed": 1234, "dataset": "hyperplanes-parity", "order-seed": 1234, "models-trained": 100, "dimension": 10, "n_hyperplanes": 2, "dim_hyperplanes": 2, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation": None},
        {"dataset":"fin-bench-cd2", "order-seed": 1234, "models-trained": 100, "gen_noise": False, "permutation":None},
        {"dataset":"fin-bench-ld1", "order-seed": 1234, "models-trained": 100, "gen_noise": False, "permutation":None},
        {"dataset":"fin-bench-cf1", "order-seed": 1234, "models-trained": 100, "gen_noise": False, "permutation":None},
        {"dataset":"fin-bench-cc3", "order-seed": 1234, "models-trained": 100, "gen_noise": False, "permutation":None},
]

vqc_params = {"max_steps": 10000, "n_layers": 5, "observable_type": "single", "convergence_interval":600}
models = [
        {"model": "drc", "n_layers": 5, "observable_type": "single", "entanglement_pattern":"block", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": np.random.randint(0, 100000)},
    {"model": "drc", "n_layers": 5, "observable_type": "full", "entanglement_pattern":"block", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": np.random.randint(0, 100000)},
    {"model": "drc", "n_layers": 5, "observable_type": "single", "entanglement_pattern": "ring", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": np.random.randint(0, 100000)},
    {"model": "drc", "n_layers": 5, "observable_type": "full", "entanglement_pattern": "ring", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": np.random.randint(0, 100000)},
]

def get_dataset(param):
    if "dataset-seed" in param:
        np.random.seed(param["dataset-seed"])

    match param["dataset"]:
        case "bars-and-stripes":
            X_data, y_data = generate_bars_and_stripes(param["n_train"]+param["n_test"], param["height"], param["width"], param["noise-std"])

            X_data, y_data = shuffle(X_data, y_data)

            X, X_test = X_data[:param["n_train"]], X_data[param["n_train"]:]
            y, y_test = y_data[:param["n_train"]], y_data[param["n_train"]:]

            filename= 'bars_and_stripes_dist'
        case "hidden-manifold":
            X_data, y_data = generate_hidden_manifold_model(param["n_train"]+param["n_test"], param["dimension"], param["manifold_dimension"])

            X_data, y_data = shuffle(X_data, y_data)

            X, X_test = X_data[:param["n_train"]], X_data[param["n_train"]:]
            y, y_test = y_data[:param["n_train"]], y_data[param["n_train"]:]

            filename= 'hidden_manifold_dist'
        case "hyperplanes-parity":
            X_data, y_data = generate_hyperplanes_parity(param["n_train"]+param["n_test"], param["dimension"], param["n_hyperplanes"], param["dim_hyperplanes"])

            X_data, y_data = shuffle(X_data, y_data)

            X, X_test = X_data[:param["n_train"]], X_data[param["n_train"]:]
            y, y_test = y_data[:param["n_train"]], y_data[param["n_train"]:]

            filename= 'hyperplanes_dist'
        case "linearly-separable":
            X_data, y_data = generate_linearly_separable(param["n_train"]+param["n_test"], param["dimension"], param["margin"])

            X_data = np.array(X_data)
            y_data = np.array(y_data)

            #X_data, y_data = shuffle(X_data, y_data)

            #X, X_test = X_data[:param["n_train"]], X_data[param["n_train"]:]
            #y, y_test = y_data[:param["n_train"]], y_data[param["n_train"]:]

            X, X_test, y, y_test = train_test_split(X_data, y_data, test_size = 0.2)
            #X_test = np.array(X_test)
            filename= 'linearly_separable_dist'
        case "pca-mnist":
            X, X_test, y, y_test = generate_mnist(0, 1, 'pca', param["dimension"], param["n_train"], param["dimension"])
            X, y = shuffle(X, y)

            filename= 'mnist_dist'
        case "two-curves":
            X_data, y_data = generate_two_curves(param["n_train"]+param["n_test"], param["n_features"], param["degree"], param["offset"], param["noise"])

            X_data, y_data = shuffle(X_data, y_data)

            X, X_test = X_data[:param["n_train"]], X_data[param["n_train"]:]
            y, y_test = y_data[:param["n_train"]], y_data[param["n_train"]:]

            filename= 'two_curves_dist'
        case "fin-bench-cd2":
            X, X_test, y, y_test = create_finbench_cd2()
        case "fin-bench-ld1":
            X, X_test, y, y_test = create_finbench_ld1()
        case "fin-bench-cf1":
            X, X_test, y, y_test = create_finbench_cf1()
        case "fin-bench-cc3":
            X, X_test, y, y_test = create_finbench_cc3()

    if param["gen_noise"]:
        X, X_test = generate_noisy_dataset(X, X_test)

    return X, y, X_test, y_test

def get_model(model_params, seed = None):
    model = None

    match(model_params["model"]):
        case "drc":
            model = DataReuploadingClassifier(
                        n_layers=model_params["n_layers"],
                        observable_type=model_params["observable_type"],
                        convergence_interval=model_params["convergence_interval"],
                        max_steps=model_params["max_steps"],
                        learning_rate=model_params["learning_rate"],
                        random_state = 42)
        case "drx":
            model = DataReuploadingRX(
                        n_layers=model_params["n_layers"],
                        observable_type=model_params["observable_type"],
                        entangling_layer=model_params["entanglement_pattern"],
                        convergence_interval=model_params["convergence_interval"],
                        max_steps=model_params["max_steps"],
                        learning_rate=model_params["learning_rate"],
                        random_state = seed)

        case "iqvc":
            model = IQPVariationalClassifier(
                        n_layers=model_params["n_layers"],
                        convergence_interval=model_params["convergence_interval"],
                        max_steps=model_params["max_steps"],
                        learning_rate=model_params["learning_rate"],
                        random_state = seed
                    )
        case "iqk":
            model = IQPKernelClassifier()
        case "aqk":
            model = AltKernelClassifier()
        case "iqdr":
            model = IQPDataReuploading(
                    n_layers=model_params["n_layers"],
                    observable_type=model_params["observable_type"],
                    convergence_interval=model_params["convergence_interval"],
                    max_steps=model_params["max_steps"],
                    learning_rate=model_params["learning_rate"],
                    random_state = seed)


    return model

def create_permutations(num_features, order_seed, symmetric = False, models_trained = 100):
    np.random.seed(order_seed)
    perms = []
    # default order
    perms.append(np.arange(num_features))

    # random orders
    while(len(perms) != models_trained):
        p = np.arange(num_features)
        np.random.shuffle(p)

        perms.append(p)

    return perms

def solve_params(data_params, model_params, random_seeds):
    X, y, X_test, y_test = get_dataset(data_params)
    perms = None

    if data_params["permutation"] is None:
        if data_params["gen_noise"]:
            perms = noise_data_perms(len(X[0]))
        else:
            symmetry = False
            if model_params["model"] == "iqp":
                symmetry = True
            perms = create_permutations(len(X[0]), data_params["order-seed"], symmetry, data_params["models-trained"])
    else:
        perms = data_params["models-trained"] * [data_params["permutation"]]

    i = 0

    test_acc = []
    train_acc = []
    results = []

    for p, i in zip(perms, range(len(perms))):
        model = get_model(model_params, None)
        X_perm = X[:, p]
        X_test_perm = X_test[:, p]

        try:
            model.fit(X_perm, y)
        except Exception as e: # Model raises a convergence error
            results.append([list(p), -100, -200])#, model.train_acc_history, model.test_acc_history, model.loss_history])
            continue

        train_acc = model.score(X_perm, y)
        test_acc = model.score(X_test_perm, y_test)

        results.append([list(p), train_acc, test_acc])#, model.train_acc_history, model.test_acc_history, model.loss_history_])

    return results

if __name__ == "__main__":
    random_seeds = np.random.randint(0, 9999999, size = 50)
    d_info = {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": 10, "dimension": 15, "manifold_dimension": 3, "gen_noise":False, "n_train": 1000, "n_test":1000}
    #model_info = {"model": "drc", "n_layers": 5, "observable_type": "full", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01}
    #model_info = {"model": "iqk"}

    for d in dataset_list:
        model_info = {"model": "drc", "n_layers": 10, "observable_type": "full", "convergence_interval": 200, "max_steps": 10000, "learning_rate": 0.01, "random_state": np.random.randint(0, 100000)}
        solve_params(d, model_info, random_seeds = random_seeds)
