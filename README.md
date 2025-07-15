# QMLPerm
Repository for research project on data permutation for quantum  machine learning. Most code related to models and datasets is from https://github.com/XanaduAI/qml-benchmarks
 related to the article "Better than classical? the subtle art of benchmarking quantum machine learning models."[1]. The experiment code is produced by us.

 [1] Bowles, Joseph, Shahnawaz Ahmed, and Maria Schuld. "Better than classical? the subtle art of benchmarking quantum machine learning models." arXiv preprint arXiv:2403.07059 (2024).

## Installation
requirements.txt file contains the important dependencies and versions. Use pyenv to create a virtual environment as many of the dependecies are a bit on the older side. The code has been ran on Linux environments, some issues with Windows machines has been noticed.

## Use
The src folder contains run_experiments.py script, which can be used to run the related experiments. Running the script will create compute results into folders for all experiments of the article. Each experiment is formulated into its own function. The models come with optimized hyperparameters for learning_rate and n_layers. Most of the results have been searched using the original qml-benchmarks library separately from this program, but also a simple grid search routine in find_hyperparams function has been defined in the run_experiments.py file.

The results of the experiments can be plotted using the various plot_scripts in the src folder and the p-values can be computed using the compute_welchs_test.py
