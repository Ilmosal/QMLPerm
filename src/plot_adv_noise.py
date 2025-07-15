import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import sys

results = None
error_bars = False

datasets = [
    'circle_single',
    'block_single',
    'circle_full',
    'block_full',
]

dataset_labels= [
        "Single Circle\n$p_S$=1.41e-29\n$p_E$=6.53e-10\n$p_R$=3.59e-10",
        "Single Block\n$p_S$=1.25e-49\n$p_E$=0.0873\n$p_R$=3.91e-08",
        "Global Circle\n$p_S$=0.0106\n$p_E$=0.0638\n$p_R$=0.7694",
        "Global Block\n$p_S$=0.0003\n$p_E$=0.0559\n$p_R$=0.1314"
]

x_labels = [
    "Default",
    "Swap",
    "Every other",
    "Random",
]

results = []

for d in datasets:
    with open("exp_results_noise/advperm_hidden-manifold_drc_{0}_results.json".format(d), 'r') as f:
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

space = 0.10

for i in range(len(results)):
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

plt.savefig("plots/adv_noise_plot.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
