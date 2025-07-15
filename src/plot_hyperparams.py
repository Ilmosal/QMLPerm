"""
File for the script for plotting the hyperparameter experiment
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import sys
import scipy

results = []

plt_labels = [
    "$P_{\\text{Def}}$",
    "$P_{1}$",
    "$P_{2}$",
    "$P_{3}$",
    "$P_{4}$",
    "$P_{5}$"
]

for i in range(1, 7):
    with open("exp_results_hyperparam/hidden-manifold_15_drc_circle_single_results_{0}.json".format(i), 'r') as f:
        results.append(json.load(f))


binned_results = [[],[],[],[],[],[]]

# Binning DO NOT COMBINE DIFFERENT ANSATZES

for r, i in zip(results, range(len(results))):
    for dp in r["results"]:
        if dp[2] != -200:
            binned_results[i].append(dp[2])
    for dp in r["results"]:
        if dp[2] != -200:
            binned_results[i].append(dp[2])

p_vals = []

for i in range(1, 6):
    result_stats = scipy.stats.ttest_ind(binned_results[0], binned_results[i], equal_var=False)
    p_vals.append(result_stats.pvalue)

for i in range(len(p_vals)):
    if p_vals[i] > 0.0001:
        plt_labels[i+1] += "\np={0:.4f}".format(p_vals[i])
    else:
        plt_labels[i+1] += "\np={0:.2G}".format(p_vals[i])

space = 0.2

for i in range(len(binned_results)):
    bplot_1 = plt.boxplot(binned_results[i], positions = [i], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")

plt.title("hidden-manifold")
plt.xticks(ticks = np.arange(6), labels=plt_labels)

plt.savefig("plots/hyperparams.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
