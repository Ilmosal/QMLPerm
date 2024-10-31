"""
Module containing a function for creating noisy data. The function adds sampled gaussian noise variables to an existing datasets to represent variables that are not useful for classification tasks for detecting where they should be located in an ansatz.
"""

import numpy as np

def generate_noisy_dataset(X, X_test, new_variable_num = None, seed = 1):
    np.random.seed(seed)

    x_len = len(X[0])

    if new_variable_num is None:
        new_variable_num = x_len

    x_size = x_len + new_variable_num

    X_new = np.random.normal(size = (len(X), x_size))
    X_test_new = np.random.normal(size = (len(X_test), x_size))

    X_new[:,:x_len] = X
    X_test_new[:,:x_len] = X_test

    return X_new, X_test_new

def noise_data_perms(d_len):
    perms = []

    # default perm
    perms.append(np.arange(d_len))

    # Perm for every other
    new_perm = np.arange(d_len)

    for i in range(int(d_len / 2)):
        new_perm[i*2] = i
        new_perm[i*2+1] = int(i + d_len/2)

    perms.append(new_perm)

    # rotation order 
    for i in range(1, 2):
        perms.append(np.roll(perms[1], -i))

    # rotation order 
    for i in range(1, 2):
        perms.append(np.roll(perms[0], int(d_len/2)))

    # Random orders for the initial N qubits
    for i in range(20):
        p = np.arange(int(d_len/ 2))
        np.random.shuffle(p)

        end = np.arange(int(d_len / 2)) + int(d_len/ 2)

        perms.append(np.concatenate((p, end), axis=None))

    return perms
