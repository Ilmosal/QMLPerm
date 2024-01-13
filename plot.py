"""
Plotting module for the QMLDataPerm
"""

    figs, axes = plt.subplots(2)
    axes[0].set_title("Objective function values")
    axes[1].set_title("Accuracy values")

    axes[0].plot(range(len(c)), c, color = "green")

    axes[1].plot(range(len(acc)), acc, color = "green")

    plt.show()
