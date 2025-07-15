import json
import scipy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

results = None
error_bars = False

datasets = [
    'linearly-separable_10',
    'hidden-manifold_10',
    'hyperplanes-parity_10',
]

plot_labels = [
    'linearly-separable',
    'hidden-manifold',
    'hyperplanes-parity',
]


results = []

for d in datasets:
    results.append([])
    with open("exp_results/{0}_iqvc_results.json".format(d), 'r') as f:
        results[-1].append(json.load(f))
    with open("exp_results/rand_{0}_iqvc_results.json".format(d), 'r') as f:
        results[-1].append(json.load(f))

p_vals = []

space = 0.2
for r in results:
    res = r[0]

    space_mod = 0

    if res["dataset"]["dataset"] == datasets[1]:
        space_mod = 1
    elif res["dataset"]["dataset"] == datasets[2]:
        space_mod = 2

    x_n, x_r = [[],[]]

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

    bplot_1 = plt.boxplot(x_n, positions = [-space+space_mod], patch_artist=True)
    bplot_2 = plt.boxplot(x_r, positions = [space +space_mod], patch_artist=True)
    bplot_1["boxes"][0].set_facecolor("blue")
    bplot_2["boxes"][0].set_facecolor("green")

for i in range(3):
    plot_labels[i] += p_vals[i]

plt.title("IQVC-model")
plt.xticks(ticks = np.arange(3), labels=plot_labels)
plt.gcf().set_size_inches(6, 4, forward=True)
plt.legend(handles=[mpatches.Patch(color="blue", label="Default"), mpatches.Patch(color="green", label="Random")], loc=4)

plt.show()
#plt.savefig("plots/iqvc_plot.tiff", bbox_inches="tight", dpi=600, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
