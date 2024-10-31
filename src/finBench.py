"""
Module containing wrappers for the FinBench datasets

Link to datasets:
https://huggingface.co/datasets/yuweiyin/FinBench
"""
import numpy as np

from datasets import load_dataset

def create_finbench_cd2():
    """
    Create normalized cd2 dataset, return data and labels for training and test sets
    amount of features: 23
    """
    cd2 = load_dataset("yuweiyin/FinBench", "cd2")

    train_set = cd2["train"] if "train" in cd2 else []
    test_set = cd2["test"] if "test" in cd2 else []

    # Compute the transformation into floats 
    X = np.zeros((train_set.num_rows, train_set[0]["num_features"]))
    y = np.zeros((train_set.num_rows))
    X_test = np.zeros((test_set.num_rows, test_set[0]["num_features"]))
    y_test = np.zeros((test_set.num_rows))

    for i in range(train_set.num_rows):
        X[i] = np.array(train_set[i]["X_ml"])
        y[i] = -1 if train_set[i]["y"] == 0 else 1

    for i in range(test_set.num_rows):
        X_test[i] = np.array(test_set[i]["X_ml"])
        y_test[i] = -1 if test_set[i]["y"] == 0 else 1

    return X, X_test, y, y_test

def create_finbench_cf1():
    """
    Create normalized cf2 dataset, return data and labels for training and test sets
    amount of features: 19
    """
    cf1 = load_dataset("yuweiyin/FinBench", "cf1")

    train_set = cf1["train"] if "train" in cf1 else []
    test_set = cf1["test"] if "test" in cf1 else []

    # Compute the transformation into floats 
    X = np.zeros((train_set.num_rows, train_set[0]["num_features"]))
    y = np.zeros((train_set.num_rows))
    X_test = np.zeros((test_set.num_rows, test_set[0]["num_features"]))
    y_test = np.zeros((test_set.num_rows))

    for i in range(train_set.num_rows):
        X[i] = np.array(train_set[i]["X_ml"])
        y[i] = -1 if train_set[i]["y"] == 0 else 1

    for i in range(test_set.num_rows):
        X_test[i] = np.array(test_set[i]["X_ml"])
        y_test[i] = -1 if test_set[i]["y"] == 0 else 1

    return X, X_test, y, y_test

def create_finbench_cc3():
    """
    Create normalized cc3 dataset, return data and labels for training and test sets
    amount of features: 21
    """
    cc3 = load_dataset("yuweiyin/FinBench", "cc3")

    train_set = cc3["train"] if "train" in cc3 else []
    test_set = cc3["test"] if "test" in cc3 else []

    # Compute the transformation into floats 
    X = np.zeros((train_set.num_rows, train_set[0]["num_features"]))
    y = np.zeros((train_set.num_rows))
    X_test = np.zeros((test_set.num_rows, test_set[0]["num_features"]))
    y_test = np.zeros((test_set.num_rows))

    for i in range(train_set.num_rows):
        X[i] = np.array(train_set[i]["X_ml"])
        y[i] = -1 if train_set[i]["y"] == 0 else 1

    for i in range(test_set.num_rows):
        X_test[i] = np.array(test_set[i]["X_ml"])
        y_test[i] = -1 if test_set[i]["y"] == 0 else 1

    return X, X_test, y, y_test

def create_finbench_ld1():
    """
    Create normalized ld1 dataset, return data and labels for training and test sets
    amount of features: 12
    """
    ld1 = load_dataset("yuweiyin/FinBench", "ld1")

    train_set = ld1["train"] if "train" in ld1 else []
    test_set = ld1["test"] if "test" in ld1 else []

    # Compute the transformation into floats 
    X = np.zeros((train_set.num_rows, train_set[0]["num_features"]))
    y = np.zeros((train_set.num_rows))
    X_test = np.zeros((test_set.num_rows, test_set[0]["num_features"]))
    y_test = np.zeros((test_set.num_rows))

    for i in range(train_set.num_rows):
        X[i] = np.array(train_set[i]["X_ml"])
        y[i] = -1 if train_set[i]["y"] == 0 else 1

    for i in range(test_set.num_rows):
        X_test[i] = np.array(test_set[i]["X_ml"])
        y_test[i] = -1 if  test_set[i]["y"] == 0 else 1

    return X, X_test, y, y_test


