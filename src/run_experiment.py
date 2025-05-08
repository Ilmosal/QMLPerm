from dask import delayed
from dask import compute
from dask.distributed import Client
import copy

import json
import numpy as np
from run_models import solve_params

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def store_results(res):
    dataset, model, results = res
    file_name = ""
    perm = ""

    if dataset["permutation"] is None:
        perm = "rand_"

    if dataset["gen_noise"]:
        perm = "advperm_"

    if "dimension" not in dataset.keys():
        dataset["dimension"] = dataset["height"] * dataset["width"]

    if model["model"] == "drc":
        file_name = "exp_results/{0}{1}_{2}_{3}_{4}_{5}_results.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'], model['entanglement_pattern'], model['observable_type'])
    elif model["model"] == "iqvc":
        file_name = "exp_results/{0}{1}_{2}_results.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'])

    result_object = {
        "dataset": dataset,
        "model": model,
        "results": results
    }

    with open(file_name, 'w') as fp:
        json.dump(result_object, fp, cls=NpEncoder)

def processDataset(dataset, model):
    required_seeds = dataset['models-trained']
    if dataset['gen_noise']:
        required_seeds *= 4

    seeds_for_models = np.random.randint(0, 9999999, size=required_seeds)
    return [dataset, model, solve_params(dataset, model, seeds_for_models)]

def find_hyperparameter_search(dataset, model, seeds_for_model):
    n_layers = [1, 5, 10, 15]
    learning_rates = [0.001, 0.01, 0.1]

    best_acc =  -1.0
    best_model = None

    model_cpy = model.copy()
    dataset_cpy = dataset.copy()
    dataset_cpy['models-trained'] = len(seeds_for_model)

    for nl in n_layers:
        for lr in learning_rates:
            model_cpy['n_layers'] = nl
            model_cpy['learning_rate'] = lr

            results = solve_params(dataset_cpy, model_cpy, seeds_for_model)

            for r in results:
                if best_acc < r[2]:
                    best_acc = r[2]
                    best_model = model_cpy().copy()

    return best_model

if __name__ == "__main__":
    list_of_delayed_functions = []
    models_trained = 1
    problem_params = [
            [
                {"dataset-seed": 1234, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "margin": 0.3, "gen_noise": False, "n_train":300, "n_test":300, "permutation":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]},
                [   ["drc", 0.001, 5, "circle", "single"],
                    ["drc", 0.1, 1, "block", "single"],
                    ["drc", 0.1, 5, "block", "full"],
                    ["drc", 0.01, 5, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "bars-and-stripes", "order-seed": 1234, "models-trained": models_trained,"height": 4, "width": 4, "noise-std":0.5, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 15, "block", "single"],
                    ["drc", 0.01, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": True, "n_train":300, "n_test":300, "permutation": None},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.01, 1, "block", "full"],
                    ["drc", 0.001, 5, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "hyperplanes-parity", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "n_hyperplanes": 2, "dim_hyperplanes": 2, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
                [   ["drc", 0.001, 10, "circle", "single"],
                    ["drc", 0.1, 5, "block", "single"],
                    ["drc", 0.001, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "margin": 0.4, "gen_noise": False, "n_train":300, "n_test":300, "permutation":None},
                [   ["iqvc", 0.01, 5],
                ],
            ],
            [
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
                [   ["iqvc", 0.001, 10],
                ],
            ],
            [
                {"dataset-seed": 1234, "dataset": "hyperplanes-parity", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "n_hyperplanes": 2, "dim_hyperplanes": 2, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation": None},
                [   ["iqvc", 0.1, 15],
                ],
            ],
    ]

    models = {
        "drc":{"model": "drc", "n_layers": 5, "observable_type": "single", "entanglement_pattern":"block", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42 },
        "iqvc": {"model": "iqvc", "n_layers": 5, "n_repeats": 1, "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42},

    }

    for dataset, model_params in problem_params:
        for m in model_params:
            model = copy.deepcopy(models[m[0]])

            if m[0] == "drc":
                model["learning_rate"] = m[1]
                model["n_layers"] = m[2]
                model["entanglement_pattern"] = m[3]
                model["observable_type"] = m[4]
            elif m[0] == "iqvc":
                model["learning_rate"] = m[1]
                model["n_layers"] = m[2]

            d_cpy = copy.deepcopy(dataset)

            list_of_delayed_functions.append(delayed(processDataset)(d_cpy, model))

    ### This starts the execution with the resources available
    results = compute(list_of_delayed_functions)

    for r in results[0]:
        store_results(r)

