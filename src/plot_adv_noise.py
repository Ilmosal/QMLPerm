import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import sys
import scipy

results = None
error_bars = False

datasets = [
    'circle_single',
    'block_single',
    'circle_full',
    'block_full',
]

dataset_labels= [
        "Single Circle",
        "Single Block",
        "Global Circle",
        "Global Block"
]

x_labels = [
    "Default",
    "Swap",
    "Every other",
    "Random",
]

results = []

for d, val in zip(datasets, range(len(datasets))):
    for i in range(4):
        with open("exp_results_noise/advperm_hidden-manifold_10_drc_{0}_results_{1}.json".format(d, 4*i+val), 'r') as f:
            results.append(json.load(f))

fig, axes = plt.subplots(1, 1)

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
    bin_id = -1
    if r['model']["entanglement_pattern"] == 'circle':
        if r['model']["observable_type"] == 'single':
            bin_id = 0
        else:
            bin_id = 2
    else:
        if r['model']["observable_type"] == 'single':
            bin_id = 1
        else:
            bin_id = 3

    for dp in r["results"]:
        if dp[0] == def_perm:
            if dp[2] != -200:
                binned_results[bin_id][0].append(dp[2])
        elif dp[0] == rotation_order:
            if dp[2] != -200:
                binned_results[bin_id][1].append(dp[2])
        elif dp[0] == every_other:
            if dp[2] != -200:
                binned_results[bin_id][2].append(dp[2])
        else:
            if dp[2] != -200:
                binned_results[bin_id][3].append(dp[2])
p_vals = []

for i in range(4):
    result_stats_1 = scipy.stats.ttest_ind(binned_results[i][0], binned_results[i][1], equal_var=False)
    result_stats_2 = scipy.stats.ttest_ind(binned_results[i][0], binned_results[i][2], equal_var=False)
    result_stats_3 = scipy.stats.ttest_ind(binned_results[i][0], binned_results[i][3], equal_var=False)

    p_vals.append([result_stats_1.pvalue, result_stats_2.pvalue, result_stats_3.pvalue])

for i in range(len(p_vals)):
    add_str = "\n"
    for a in range(3):
        match a:
            case 0:
                add_str += "$p_S$="
            case 1:
                add_str += "$p_E$="
            case 2:
                add_str += "$p_R$="

        if p_vals[i][a] > 0.0001:
            add_str += "{0:.4f}\n".format(p_vals[i][a])
        else:
            add_str += "{0:.2G}\n".format(p_vals[i][a])

    dataset_labels[i] += add_str

space = 0.10

for i in range(len(binned_results)):
    bplot_1 = plt.boxplot(binned_results[i][0], positions = [i-3*space], patch_artist=True)
    bplot_2 = plt.boxplot(binned_results[i][1], positions = [i-space], patch_artist=True)
    bplot_3 = plt.boxplot(binned_results[i][2], positions = [i+space], patch_artist=True)
    bplot_4 = plt.boxplot(binned_results[i][3], positions = [i+3*space], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")
    bplot_2["boxes"][0].set_facecolor("black")
    bplot_3["boxes"][0].set_facecolor("yellow")
    bplot_4["boxes"][0].set_facecolor("green")

#plt.subplots_adjust(hspace=0.3)

plt.title("Adversarial Permutation")
plt.xticks(ticks = np.arange(4), labels=dataset_labels)
plt.gcf().set_size_inches(6, 4, forward=True)
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="black", label="Swapped"), mpatches.Patch(color="yellow", label="Every Other"), mpatches.Patch(color="green", label="Random")], loc=4)

#plt.show()
plt.savefig("plots/adv_noise_plot.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
