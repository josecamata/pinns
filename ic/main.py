import deepxde as dde
from deepxde.backend import tf
import numpy as np
from network import PINN
import matplotlib.pyplot as plt
from animation import animate_solution

k = 10 ** -8

# pde, top, right, down, left, ic
LOSS_WEIGHTS = [
    10, 
    0.1,
    0.1,
    0.1,
    0.1,
    5, 
]

def pde(X, U):
    dU_x = dde.grad.jacobian(U, X, j=0)
    dU_y = dde.grad.jacobian(U, X, j=1)
    dU_t = dde.grad.jacobian(U, X, j=2)

    dU_xx = dde.grad.hessian(U, X, i=0, j=0)
    dU_yy = dde.grad.hessian(U, X, i=1, j=1)

    return dU_t + (5 - X[:, 1:2]) * dU_x + (X[:, 0:1] - 5) * dU_y - k * (dU_xx + dU_yy)

def func_initial_condition(x):
    r = np.power(x[:, 0:1] - 5, 2) + np.power(x[:, 1:2] - 7.5, 2)
    return np.exp(-0.5 * r)

pinn = PINN(([3] + [60] * 5 + [1]), 'tanh', 'Glorot uniform', 'adam')
pinn.define_geometry(10, 10, 0, 2 * np.pi)
pinn.define_pde(pde)
pinn.define_boundaries(0, 0, 0, 0)
pinn.define_initial_condition(func_initial_condition)
pinn.training_data(3000)
pinn.train(1e-3, LOSS_WEIGHTS, 15000, 32, "loss_pulse_gaussian")

fig, ax = plt.subplots()
ax = fig.add_subplot(111)

nelx = 100  # Número de elementos em x
nely = 100  # Número de elementos em y
timesteps = 401

x = np.linspace(0, 10, nelx + 1)
y = np.linspace(0, 10, nely + 1)
t = np.linspace(0, 2 * np.pi, timesteps)

# Dados pra serem usados na predição
test_x, test_y, test_t = np.meshgrid(x, y, t)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

# Predição da solução
predicted_solution = pinn.model.predict(test_domain)
predicted_solution = predicted_solution.reshape(
    test_x.shape
)

animate_solution(
    predicted_solution,
    10,
    10,
    f"gaussian_pulse_rotation.mp4",
    "Gaussian Pulse Rotation",
    "u(x,y,t)",
    t,
)