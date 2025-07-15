"""
Script for using scipy to compute the p-values for the different probability distributions of the experiment
"""

import scipy
import sys
import json
import numpy as np

def get_statistics(dataset_choice):

    datasets = [
        ['exp_results_drc', 'bars-and-stripes'],
        ['exp_results_drc', 'hidden-manifold'],
        ['exp_results_drc', 'hyperplanes-parity'],
        ['exp_results_drc', 'linearly-separable'],
        ['exp_results_drc', 'fin-bench-cd2_23'],
        ['exp_results_iqvc', 'hidden-manifold'],
        ['exp_results_iqvc', 'linearly-separable'],
        ['exp_results_iqvc', 'hyperplanes-parity'],
        ['exp_results_diff', 'hidden_manifold'],
        ['exp_results_hyperparams', 'hidden_manifold'],
    ]

    results = []

    dir_name, d = datasets[dataset_choice]


    match dir_name:
        case 'exp_results_drc':
            for obs in ["full", "single"]:
                for ent in ["circle", "block"]:
                    results.append([])
                    with open("{3}/{0}_drc_{1}_{2}_results.json".format(d, ent, obs, dir_name), 'r') as f:
                        results[-1].append(json.load(f))
                    with open("{3}/rand_{0}_drc_{1}_{2}_results.json".format(d, ent, obs, dir_name), 'r') as f:
                        results[-1].append(json.load(f))

        case 'exp_results_iqvc':
            results.append([])
            with open("exp_results_iqvc/{0}_iqvc_results.json".format(d), 'r') as f:
                results[-1].append(json.load(f))
            with open("exp_results_iqvc/rand_{0}_iqvc_results.json".format(d), 'r') as f:
                results[-1].append(json.load(f))
        case 'exp_results_diff':
            for i in [15,18,21,24,27,30]:
                results.append([])
                with open("exp_results_diff/hidden-manifold_{1}_drc_circle_full_results.json".format(d, i), 'r') as f:
                    results[-1].append(json.load(f))
                with open("exp_results_diff/rand_hidden-manifold_{1}_drc_circle_full_results.json".format(d, i), 'r') as f:
                    results[-1].append(json.load(f))

    for r in results:
        res = r[0]
        x_n, x_r = [[],[]]

        for dp in r[0]["results"]:
            if dp[-1] > 0.0:
                x_n.append(dp[-1])

        for dp in r[1]["results"]:
            if dp[-1] > 0.0:
                x_r.append(dp[-1])

        result_stats = scipy.stats.ttest_ind(x_n, x_r, equal_var=False)

        match dir_name:
            case 'exp_results_drc':
                print(r[0]["model"]["observable_type"] + " - " + r[0]["model"]["entanglement_pattern"])
            case 'exp_results_iqvc':
                print("IQVC-model")
            case 'exp_results_diff':
                print("DRC - model size: {0}".format(r[0]["dataset"]["dimension"]))

        print("Statistic: " + str(result_stats.statistic))
        print("P-value: " + str(result_stats.pvalue))
        print("Degrees of freedom: " + str(result_stats.df))
        print("")

def get_adv_perm_statistics():
    datasets = [
        'circle_single',
        'block_single',
        'circle_full',
        'block_full',
    ]

    results = []

    for d in datasets:
        with open("exp_results_noise/advperm_hidden-manifold_drc_{0}_results.json".format(d), 'r') as f:
            results.append(json.load(f))


    dim = int(2 * results[0]["dataset"]["dimension"])
    def_perm = list(np.arange(dim))
    rotation_order = list(np.roll(np.arange(20), int(dim/2)))
    every_other = list(np.zeros(dim))

    for i in range(int(dim/2)):
        every_other[i*2] = int(i)
        every_other[i*2+1] = int(i+dim/2)

    binned_results = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]

    # Binning DO NOT COMBINE DIFFERENT ANSATZES
    for r, i in zip(results, range(len(results))):
        for dp in r["results"]:
            if dp[0] == def_perm:
                if dp[2] != -200:
                    binned_results[i][0].append(dp[2])
            elif dp[0] == rotation_order:
                if dp[2] != -200:
                    binned_results[i][1].append(dp[2])
            elif dp[0] == every_other:
                if dp[2] != -200:
                    binned_results[i][2].append(dp[2])
            else:
                if dp[2] != -200:
                    binned_results[i][3].append(dp[2])

    for r, i in zip(binned_results, range(len(binned_results))):
        result_stats_swp = scipy.stats.ttest_ind(r[0], r[1], equal_var=False)
        result_stats_alt = scipy.stats.ttest_ind(r[0], r[2], equal_var=False)
        result_stats_rand = scipy.stats.ttest_ind(r[0], r[3], equal_var=False)

        print("Ansatz type: " + datasets[i])
        print("SWAP")
        print("Statistic: " + str(result_stats_swp.statistic))
        print("P-value: " + str(result_stats_swp.pvalue))
        print("Degrees of freedom: " + str(result_stats_swp.df))
        print("ALTERNATE")
        print("Statistic: " + str(result_stats_alt.statistic))
        print("P-value: " + str(result_stats_alt.pvalue))
        print("Degrees of freedom: " + str(result_stats_alt.df))
        print("RANDOM")
        print("Statistic: " + str(result_stats_rand.statistic))
        print("P-value: " + str(result_stats_rand.pvalue))
        print("Degrees of freedom: " + str(result_stats_rand.df))
        print("")

if __name__ == "__main__":
    #get_statistics(int(sys.argv[1]))
    get_adv_perm_statistics()
