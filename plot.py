"""
Plotting module for the QMLDataPerm
"""   
import json
import matplotlib.pyplot as plt
import os

directory = 'results'
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    with open(file) as json_file:
        contents = json.load(json_file)
        values = list(contents.values())
        acc = [x[0] for x in values]
        c = [x[1] for x in values]
        x = file[8:-5]
        figs, axes = plt.subplots(2)
        axes[0].set_title("Objective function values")
        axes[1].set_title("Accuracy values")

        axes[0].plot(range(len(c)), c, color = "green")

        axes[1].plot(range(len(acc)), acc, color = "green")
        if not os.path.exists('plots'):
            os.mkdir('plots')
        s = os.path.dirname(os.path.abspath(__file__)) + r"\plots\\" + f'{x}.png'
        plt.savefig(s)
        plt.show()