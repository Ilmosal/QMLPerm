import sys
import json

import scipy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

dataset_choice = int(sys.argv[1])

results = None
error_bars = False

datasets = [
    'bars-and-stripes_16',
    'hidden-manifold_15',
    'hyperplanes-parity_15',
    'linearly-separable_15',
]

plot_labels = [
    'bars-and-stripes',
    'hidden-manifold',
    'hyperplanes-parity',
    'linearly-separable',
]

# single circle, single block, full circle, full block
results = []

d = datasets[dataset_choice]
for obs in ["single", "full"]:
    for ent in ["circle", "block"]:
        results.append([])
        with open("exp_results/{0}_drc_{1}_{2}_results.json".format(d, ent, obs), 'r') as f:
            results[-1].append(json.load(f))
        with open("exp_results/rand_{0}_drc_{1}_{2}_results.json".format(d, ent, obs), 'r') as f:
            results[-1].append(json.load(f))


def params_to_i(obs, ent):
    if obs == "single" and ent == "circle":
        return 0
    if obs == "single" and ent == "block":
        return 1
    if obs == "full" and ent == "circle":
        return 2
    if obs == "full" and ent == "block":
        return 3

space = 0.2
p_vals = []

for r in results:
    res = r[0]
    x_n, x_r = [[],[]]
    i = params_to_i(res["model"]["observable_type"], res["model"]["entanglement_pattern"])

    for dp in res["results"]:
        if dp[-1] > 0.0:
            x_n.append(dp[-1])
    for dp in r[1]["results"]:
        if dp[-1] > 0.0:
            x_r.append(dp[-1])

    result_stats = scipy.stats.ttest_ind(x_n, x_r, equal_var=False)

    if result_stats.pvalue > 0.0001:
        p_vals.append("\np={0:.4f}".format(result_stats.pvalue))
    else:
        p_vals.append("\np={0:.2G}".format(result_stats.pvalue))

    bplot_1 = plt.boxplot(x_n, positions = [i-space], patch_artist=True)
    bplot_2 = plt.boxplot(x_r, positions = [i+space], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")
    bplot_2["boxes"][0].set_facecolor("green")

#axes[0,0].set_title(datasets[0])
plt.xticks(ticks = np.arange(4), labels=["Single Circle\np = {0}".format(p_vals[0]), "Single Block\np = {0}".format(p_vals[1]), "Global Circle\np = {0}".format(p_vals[2]), "Global Block\np = {0}".format(p_vals[3])])
plt.gcf().set_size_inches(6, 4, forward=True)
plt.title(plot_labels[dataset_choice])
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="green", label="Random")], loc=4)

#plt.show()
plt.savefig("plots/{0}_results_plot.tiff".format(datasets[dataset_choice]), bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
