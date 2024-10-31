import numpy as np

from data.bars_and_stripes import generate_bars_and_stripes
from data.hidden_manifold import generate_hidden_manifold_model
from data.hyperplanes import generate_hyperplanes_parity
from data.linearly_separable import generate_linearly_separable
from data.mnist import generate_mnist
from data.two_curves import generate_two_curves

from vqc.tsp import solve


for i in range(6):
    for j in range(8, 13):

        match i:
            case 0:
                X, y = generate_bars_and_stripes(400, j, j, 0.5)
                X = X.reshape(len(X), len(X[0][0]) ** 2)
                filename= 'bars_and_stripes_dist'
            case 1:
                X, y = generate_hidden_manifold_model(400, j, 3)
                filename= 'hidden_manifold_dist'
            case 2:
                X, y = generate_hyperplanes_parity(400, j, 3, 3)
                filename= 'hyperplanes_dist'
            case 3:
                X, y = generate_linearly_separable(400, j, 0.5)
                X = np.array(X)
                filename= 'linearly_separable_dist'
            case 4:
                X_train, X_test, y_train, y_test = generate_mnist(0, 1, 'pca', j, 400, j)
                X = X_train
                filename= 'mnist_dist'
            case 5:
                X, y = generate_two_curves(400, j, 3, 0.5, 0.5)
                filename= 'two_curves_dist'
        f = open(f'{filename}.txt', 'a')

        if j == 8:
            f.write('\n')
            f.write('Iteration 3\n')
        var_mat = np.abs(np.corrcoef(X[:,:], rowvar=False))
        min_perm, min_dist = solve(var_mat, linear = False)
        max_perm, max_dist = solve(var_mat, maximize = True, linear = False)
        f.write(f"Dimension: {j}, min_dist: {min_dist}, max_dist: {max_dist}\n")
        f.close()


