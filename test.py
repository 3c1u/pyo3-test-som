from ddkdk import Som
import numpy as np

som = Som(2, 4)

x, y = np.meshgrid(np.arange(0, 1, 1/6), np.arange(0, 1, 1/6))
grid = np.vstack([x.ravel(), y.ravel()])
grid = grid.T

input_data = np.array(
    [[ 0.80,  0.55,  0.22,  0.03],
    [ 0.82,  0.50,  0.23,  0.03],
    [ 0.80,  0.54,  0.22,  0.03],
    [ 0.80,  0.53,  0.26,  0.03],
    [ 0.79,  0.56,  0.22,  0.03],
    [ 0.75,  0.60,  0.25,  0.03],
    [ 0.77,  0.59,  0.22,  0.03]]
)

som.fit(input_data, grid, max_iter=20, tau=10.0)

print(som.latent)
