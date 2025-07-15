import sys
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

dataset_choice = int(sys.argv[1])

results = None
error_bars = False

datasets = [
    'bars-and-stripes',
    'hidden-manifold',
    'hyperplanes-parity',
    'linearly-separable',
    'fin-bench-cd2_23'
]

dataset_labels = [
    'bars-and-stripes',
    'hidden-manifold',
    'hyperplanes-parity',
    'linearly-separable',
    'fin-bench-cd2'
]

# single circle, single block, full circle, full block
p_values = {
    'bars-and-stripes': ['2.69e-05', '5.15e-10', '0.0065', '6.43e-06'],
    'hidden-manifold': ['1.65e-10', '1.02e-06', '2.04e-15', '0.3616'],
    'hyperplanes-parity': ['5.97e-17', '9.69e-08', '1.67e-16', '1.37e-16'],
    'linearly-separable': ['0.3819','2.45e-08', '0.2397', '0.4176'],
    'fin-bench-cd2_23': ['0.0446', '0.0624', '0.0003', '2.42e-05']
}

results = []

d = datasets[dataset_choice]
for obs in ["full", "single"]:
    for ent in ["circle", "block"]:
        results.append([])
        with open("exp_results_drc/{0}_drc_{1}_{2}_results.json".format(d, ent, obs), 'r') as f:
            results[-1].append(json.load(f))
        with open("exp_results_drc/rand_{0}_drc_{1}_{2}_results.json".format(d, ent, obs), 'r') as f:
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

    bplot_1 = plt.boxplot(x_n, positions = [i-space], patch_artist=True)
    bplot_2 = plt.boxplot(x_r, positions = [i+space], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")
    bplot_2["boxes"][0].set_facecolor("green")

#axes[0,0].set_title(datasets[0])
plt.xticks(ticks = np.arange(4), labels=["Single Circle\np = {0}".format(p_values[d][0]), "Single Block\np = {0}".format(p_values[d][1]), "Global Circle\np = {0}".format(p_values[d][2]), "Global Block\np = {0}".format(p_values[d][3])])
plt.gcf().set_size_inches(6, 4, forward=True)
plt.title(dataset_labels[dataset_choice])
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="green", label="Random")], loc=4)


plt.savefig("plots/{0}_results_plot.tiff".format(datasets[dataset_choice]), bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
