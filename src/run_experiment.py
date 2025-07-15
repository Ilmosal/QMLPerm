from dask import delayed
from dask import compute
from dask.distributed import Client

import copy
import os
import sys

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

def store_results(res, folder_name, file_num = None):
    dataset, model, results = res
    file_name = ""
    perm = ""
    numb_text = ""

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if dataset["permutation"] is None:
        perm = "rand_"

    if dataset["gen_noise"]:
        perm = "advperm_"

    if "dimension" not in dataset.keys():
        dataset["dimension"] = dataset["height"] * dataset["width"]

    if file_num != None:
        numb_text += "_" + str(file_num)

    if model["model"] == "drc":
        file_name = "{6}/{0}{1}_{2}_{3}_{4}_{5}_results{7}.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'], model['entanglement_pattern'], model['observable_type'], folder_name, numb_text)
    elif model["model"] == "iqvc":
        file_name = "{4}/{0}{1}_{2}_{3}_results{5}.json".format(perm, dataset['dataset'], dataset["dimension"], model['model'], folder_name, numb_text)

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

    np.random.seed(model['random_state'])
    seeds_for_models = np.random.randint(0, 9999999, size=required_seeds)

    return [dataset, model, solve_params(dataset, model, seeds_for_models)]

def find_hyperparams(dataset, model, seeds_for_model):
    n_layers = [5, 10, 15]
    learning_rates = [0.001, 0.01, 0.1]

    required_seeds = dataset['models-trained']
    np.random.seed(model['random_state'])
    seeds_for_models = np.random.randint(0, 9999999, size=required_seeds)

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

            model_results.append([model_cpy.copy(), avg_result, var_result, max_result])

    return model_results, best_acc, best_model

def run_hyperparameter_experiment(models_trained):
    list_of_delayed_functions = []
    problem_params = [
    [
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]},
        [   ["drc", 0.01, 5, "circle", "single"],
        ]
    ],
[
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [4, 11, 1, 7, 13, 10, 12, 14, 2, 5, 3, 8, 6, 9, 0]},
        [   ["drc", 0.01, 10, "circle", "single"],
        ]
    ],[
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [3, 9, 11, 10, 8, 7, 2, 13, 12, 1, 6, 14, 4, 5, 0]},
        [   ["drc", 0.01, 5, "circle", "single"],
        ]
    ],[
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [9, 12, 11, 14, 3, 2, 13, 4, 10, 8, 7, 5, 0, 6, 1]},
        [   ["drc", 0.01, 5, "circle", "single"],
        ]
    ],[
        {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [14, 3, 0, 13, 1, 9, 5, 6, 2, 4, 12, 11, 7, 10, 8]},
        [   ["drc", 0.01, 5, "circle", "single"],
        ]
    ],[
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": [6, 1, 11, 5, 9, 8, 0, 13, 14, 3, 7, 10, 12, 4, 2]},
               [   ["drc", 0.01, 5, "circle", "single"],
                ]
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

    results = compute(list_of_delayed_functions)

    for r, i in zip(results[0], range(len(results[0]))):
        store_results(r, "exp_results_hyperparam", i+1)

def run_diff_experiment(models_trained):
    list_of_delayed_functions = []
    problem_params = [
        [
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 18, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 21, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 24, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 27, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 30, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(15))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 18, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(18))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 21, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(21))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 24, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(24))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 27, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(27))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
        ],[
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 30, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(30))},
            [   ["drc", 0.01, 5, "circle", "single"],
            ]
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

    results = compute(list_of_delayed_functions)

    for r in results[0]:
        store_results(r, "exp_results_diff")

def run_adv_perm_experiment(models_trained):
    list_of_delayed_functions = []

    def_perm = list(range(20))

    # Perm for every other
    every_other = list(range(20))

    for i in range(10):
        every_other[i*2] = i
        every_other[i*2+1] = i + 10

    # rotation order 
    rot_perm = list(range(20))

    for i in range(20):
        rot_perm[i] = (i + 10) % 20

    problem_params = [
        [
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": True, "n_train":300, "n_test":300, "permutation": None},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.001, 5, "circle", "full"],
                    ["drc", 0.01, 5, "block", "full"],
            ]
        ],
        [
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": True, "n_train":300, "n_test":300, "permutation": def_perm},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.001, 5, "circle", "full"],
                    ["drc", 0.01, 5, "block", "full"],
            ]
        ],
        [
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": True, "n_train":300, "n_test":300, "permutation": rot_perm},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.001, 5, "circle", "full"],
                    ["drc", 0.01, 5, "block", "full"],
            ]
        ],
        [
            {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": True, "n_train":300, "n_test":300, "permutation": every_other},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.001, 5, "circle", "full"],
                    ["drc", 0.01, 5, "block", "full"],
            ]
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

    results = compute(list_of_delayed_functions)

    for r, i in zip(results[0], range(len(results[0]))):
        store_results(r, "exp_results_noise", i)

def run_model_experiments(models_trained):
    list_of_delayed_functions = []
    problem_params = [
            [
                {"dataset-seed": 1234, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "margin": 0.3, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(15))},
                [   ["drc", 0.001, 5, "circle", "single"],
                    ["drc", 0.1, 1, "block", "single"],
                    ["drc", 0.1, 5, "block", "full"],
                    ["drc", 0.01, 5, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "bars-and-stripes", "order-seed": 1234, "models-trained": models_trained,"height": 4, "width": 4, "noise-std":0.5, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation": list(range(16))},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 15, "block", "single"],
                    ["drc", 0.01, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(15))},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 10, "block", "single"],
                    ["drc", 0.01, 1, "block", "full"],
                    ["drc", 0.001, 5, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "hyperplanes-parity", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "n_hyperplanes": 2, "dim_hyperplanes": 2, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(15))},
                [   ["drc", 0.001, 10, "circle", "single"],
                    ["drc", 0.1, 5, "block", "single"],
                    ["drc", 0.001, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "margin": 0.4, "gen_noise": False, "n_train":300, "n_test":300, "permutation":list(range(10))},
                [   ["iqvc", 0.01, 5],
                ],
            ],
            [
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": list(range(10))},
                [   ["iqvc", 0.001, 10],
                ],
            ],
            [
                {"dataset-seed": 1234, "dataset": "hyperplanes-parity", "order-seed": 1234, "models-trained": models_trained, "dimension": 10, "n_hyperplanes": 2, "dim_hyperplanes": 2, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation": list(range(10))},
                [   ["iqvc", 0.1, 15],
                ],
            ],
            [
                {"dataset-seed": 1234, "dataset": "linearly-separable", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "margin": 0.3, "gen_noise": False, "n_train":300, "n_test":300, "permutation":None},
                [   ["drc", 0.001, 5, "circle", "single"],
                    ["drc", 0.1, 1, "block", "single"],
                    ["drc", 0.1, 5, "block", "full"],
                    ["drc", 0.01, 5, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "bars-and-stripes", "order-seed": 1234, "models-trained": models_trained,"height": 4, "width": 4, "noise-std":0.5, "gen_noise": False, "n_train":1000, "n_test":1000, "permutation":None},
                [   ["drc", 0.01, 5, "circle", "single"],
                    ["drc", 0.01, 15, "block", "single"],
                    ["drc", 0.01, 15, "block", "full"],
                    ["drc", 0.01, 15, "circle", "full"],
                ]
            ],
            [
                {"dataset-seed": 1234, "dataset": "hidden-manifold", "order-seed": 1234, "models-trained": models_trained, "dimension": 15, "manifold_dimension": 6, "gen_noise": False, "n_train":300, "n_test":300, "permutation": None},
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

    results = compute(list_of_delayed_functions)

    for r in results[0]:
        store_results(r, "exp_results")

if __name__ == "__main__":
    try:
        exp_id = int(sys.argv[1])
        n_models = int(sys.argv[2])

        if n_models <= 0 or n_models > 10000:
            raise Exception("Invalid n_models: {0}".format(n_models))

    except Exception as e:
        print("Faulty arguments for the program, give experiment id from 0 to 3 and then the amount of models")
        print("Exception: {0}".format(e))
    match exp_id:
        case 0:
            run_model_experiments(n_models) # Run the experiments for Figures 5 and 6a
        case 1:
            run_diff_experiment(n_models) # Run Figure 6b experiment
        case 2:
            run_adv_perm_experiment(n_models) # Run Figure 7 experiment
        case 3:
            run_hyperparameter_experiment(n_models) # Run Figure 6c experiment
