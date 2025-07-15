import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import sys

results = None
results = []

plot_labels = [
    '15\np =\n1.75e-18',
    '18\np =\n1.11e-21',
    '21\np =\n1.82e-10',
    '24\np =\n5.71e-17',
    '27\np =\n5.09e-05',
    '30\np =\n0.1606',
]

for i in range(15, 31, 3):
    results.append([])
    with open("exp_results_diff_2/hidden-manifold_{0}_drc_circle_single_results.json".format(i), 'r') as f:
        results[-1].append(json.load(f))
    with open("exp_results_diff_2/rand_hidden-manifold_{0}_drc_circle_single_results.json".format(i), 'r') as f:
        results[-1].append(json.load(f))

binned_results = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

# Binning DO NOT COMBINE DIFFERENT ANSATZES
for r, i in zip(results, range(len(results))):
    for dp in r[0]["results"]:
        if dp[2] != -200:
            binned_results[i][0].append(dp[2])
    for dp in r[1]["results"]:
        if dp[2] != -200:
            binned_results[i][1].append(dp[2])

space = 0.2

for i in range(len(binned_results)):
    bplot_1 = plt.boxplot(binned_results[i][0], positions = [i-space], patch_artist=True)
    bplot_2 = plt.boxplot(binned_results[i][1], positions = [i+space], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")
    bplot_2["boxes"][0].set_facecolor("green")

plt.title("hidden-manifold")
plt.xticks(ticks = np.arange(6), labels=plot_labels)
plt.gcf().set_size_inches(6, 4, forward=True)
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="green", label="Random")], loc=4)

plt.savefig("plots/diff_plot.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
