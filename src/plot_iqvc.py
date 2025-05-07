import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

results = None
error_bars = False

datasets = [
    'linearly-separable',
    'hidden-manifold',
    'hyperplanes-parity',
]

results = []

for d in datasets:
    results.append([])
    with open("exp_results_iqvc/{0}_iqvc_results.json".format(d), 'r') as f:
        results[-1].append(json.load(f))
    with open("exp_results_iqvc/perm_{0}_iqvc_results.json".format(d), 'r') as f:
        results[-1].append(json.load(f))
space = 0.2
for r in results:
    res = r[0]

    if res["dataset"]["dataset"] == datasets[0]:
        x_n, x_r = [[],[]]

        for dp in res["results"]:
            if dp[-1] > 0.0:
                x_n.append(dp[-1])
        for dp in r[1]["results"]:
            if dp[-1] > 0.0:
                x_r.append(dp[-1])

        bplot_1 = plt.boxplot(x_n, positions = [-space], patch_artist=True)
        bplot_2 = plt.boxplot(x_r, positions = [space], patch_artist=True)
        bplot_1["boxes"][0].set_facecolor("blue")
        bplot_2["boxes"][0].set_facecolor("green")

    elif res["dataset"]["dataset"] == datasets[1]:
        x_n, x_r = [[],[]]

        for dp in res["results"]:
            if dp[-1] > 0.0:
                x_n.append(dp[-1])
        for dp in r[1]["results"]:
            if dp[-1] > 0.0:
                x_r.append(dp[-1])

        bplot_1 = plt.boxplot(x_n, positions = [1-space], patch_artist=True)
        bplot_2 = plt.boxplot(x_r, positions = [1+space], patch_artist=True)
        bplot_1["boxes"][0].set_facecolor("blue")
        bplot_2["boxes"][0].set_facecolor("green")

    elif res["dataset"]["dataset"] == datasets[2]:
        x_n, x_r = [[],[]]

        for dp in res["results"]:
            if dp[-1] > 0.0:
                x_n.append(dp[-1])
        for dp in r[1]["results"]:
            if dp[-1] > 0.0:
                x_r.append(dp[-1])

        bplot_1 = plt.boxplot(x_n, positions = [2-space], patch_artist=True)
        bplot_2 = plt.boxplot(x_r, positions = [2+space], patch_artist=True)
        bplot_1["boxes"][0].set_facecolor("blue")
        bplot_2["boxes"][0].set_facecolor("green")

plt.title("IQVC-model")
plt.xticks(ticks = np.arange(3), labels=datasets)
plt.gcf().set_size_inches(6, 4, forward=True)
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="green", label="Random")], loc=4)

plt.savefig("plots/iqvc_plot.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
