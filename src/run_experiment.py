from dask import delayed
from dask import compute
from dask.distributed import Client
import copy
import os

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

def store_results(res, folder_name):
    dataset, model, results = res
    file_name = ""
    perm = ""

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if dataset["permutation"] is None:
        perm = "rand_"

    if dataset["gen_noise"]:
        perm = "advperm_"

    if "dimension" not in dataset.keys():
        dataset["dimension"] = dataset["height"] * dataset["width"]

    if model["model"] == "drc":
        file_name = "{6}/{0}{1}_{2}_{3}_{4}_{5}_results.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'], model['entanglement_pattern'], model['observable_type'], folder_name)
    elif model["model"] == "iqvc":
        file_name = "{6}/{0}{1}_{2}_results.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'], folder_name)

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
    n_layers = [5, 10, 15]
    learning_rates = [0.001, 0.01, 0.1]

    best_acc =  -1.0
    best_model = None

    model_cpy = model.copy()
    dataset_cpy = dataset.copy()
    dataset_cpy['models-trained'] = len(seeds_for_model)

    model_results = []

    for nl in n_layers:
        for lr in learning_rates:
            model_cpy['n_layers'] = nl
            model_cpy['learning_rate'] = lr

            results = solve_params(dataset_cpy, model_cpy, seeds_for_model)
            res = []

            for r in results:
                res.append(r[2])

            avg_result = np.average(res)
            var_result = np.var(res)
            max_result = np.max(res)

            if max_result > best_acc:
                best_acc = max_result
                best_model = model_cpy.copy()

            model_results.append([model_cpy, avg_result, var_result, max_result])

    return model_results, best_acc, best_model

def run_real_world_data_experiment():
    list_of_delayed_functions = []
    models_trained = 50
    problem_params = [
            {"dataset-seed": 1234, "dataset": "fin-bench-cd2", "order-seed": 1234, "models-trained": models_trained, "gen_noise": False, "permutation": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], "dimension":23},
            [   ["drc", 0.01, 10, "circle", "single"],
                ["drc", 0.01, 10, "block", "single"],
                ["drc", 0.01, 10, "block", "full"],
                ["drc", 0.01, 10, "circle", "full"],
            ]
    ]

    models = {
        "drc":{"model": "drc", "n_layers": 5, "observable_type": "single", "entanglement_pattern":"block", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42 },
        "iqvc": {"model": "iqvc", "n_layers": 5, "n_repeats": 1, "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42},

    }

    for m in problem_params[1]:
        model = copy.deepcopy(models[m[0]])

        if m[0] == "drc":
            model["learning_rate"] = m[1]
            model["n_layers"] = m[2]
            model["entanglement_pattern"] = m[3]
            model["observable_type"] = m[4]
        elif m[0] == "iqvc":
            model["learning_rate"] = m[1]
            model["n_layers"] = m[2]

        d_cpy = copy.deepcopy(problem_params[0])

        list_of_delayed_functions.append(delayed(processDataset)(d_cpy, model))

    results = compute(list_of_delayed_functions)

    for r in results[0]:
        store_results(r, "exp_results_normal_perm")

def run_model_experiments():
    list_of_delayed_functions = []
    models_trained = 50
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
            [
                {"dataset-seed": 1234, "dataset": "fin-bench-cd2", "order-seed": 1234, "models-trained": models_trained, "gen_noise": False, "permutation": None},
                [   ["drc", 0.001, 10, "circle", "single"],
                    ["drc", 0.1, 5, "block", "single"],
                    ["drc", 0.001, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
    ]

    models = {
        "drc":{"model": "drc", "n_layers": 5, "observable_type": "single", "entanglement_pattern":"block", "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42 },
        "iqvc": {"model": "iqvc", "n_layers": 5, "n_repeats": 1, "convergence_interval": 600, "max_steps": 10000, "learning_rate": 0.01, "random_state": 42},

    }

    seeds_for_models = [16325,1451245,531631,325716,13516,1245136,14361624,145215,134512,1352215,135314,31562326,41253145,31463146,13463146,13463146,31461367,4367358,964787,2856767]

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

            list_of_delayed_functions.append(delayed(find_hyperparameter_search)(d_cpy, model, seeds_for_models))

    results = compute(list_of_delayed_functions)

    for r in results[0]:
        store_results(r, "exp_results")

if __name__ == "__main__":
    run_real_world_data_experiment()
