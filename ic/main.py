import deepxde as dde
import numpy as np
from network import PINN
import matplotlib.pyplot as plt
from animation import animate_solution
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
import os

np.int = int

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf
    sin = tf.sin

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

def neural_network():

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

    pinn = PINN('Glorot uniform', 'adam')
    pinn.define_geometry(10, 10, 0, 2 * np.pi)
    pinn.define_pde(pde)
    pinn.define_boundaries(0, 0, 0, 0)
    pinn.define_initial_condition(func_initial_condition)
    pinn.training_data(3000)

    return pinn

pinn = neural_network()

# HPO configurações
n_calls = 11 #numero de chamadas
dim_learning_rate = Categorical(categories=[1e-4, 1.68e-4, 2.83e-4, 4.78e-4, 8.06e-4, 1e-3, 2.3e-3, 3.89e-3, 6.58e-3, 8.4e-3, 1e-2, 1.87e-2, 3.16e-2, 5e-2], name="learning_rate")
dim_num_dense_layers = Integer(low=3, high=10, name="num_dense_layers")
dim_num_dense_nodes = Categorical(categories=[20, 30, 40, 50, 60, 70, 80], name="num_dense_nodes")
dim_activation = Categorical(categories=["ReLU", "sigmoid", "tanh", "Swish", "sin"], name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
]

#Parametros padrão
default_parameters = [1e-3, 5, 60, "tanh"]

#dados de predição
num_values = 100
    
x = np.ones((num_values, 1)) * 5
y = np.linspace(0, 10, num_values).reshape(-1, 1)
t = np.ones((num_values, 1)) * (2 * np.pi)

input_data = np.hstack((x, y, t))

fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.set_xlabel('y')
ax1.set_ylabel('u(x = 5, y, t = 2π)')

results = np.zeros((n_calls, 6), dtype=object)

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
    global ITERATION

    print("\n " + str(ITERATION) + " it number")
    
    # Print the hyper-parameters.
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print()

    # Create the neural network with these hyper-parameters.
    model = pinn.create_model(config, LOSS_WEIGHTS)
    # possibility to change where we save
    error, training_time = pinn.train_model(model, 100, 32, iteration_step = ITERATION)
    # print(accuracy, 'accuracy is')

    file_name = f'outputs/loss/loss_{ITERATION}.dat'

    with open(file_name, 'r') as file:
        original_content = file.read()

    # Salva os dados usados de hiper parametros
    new_header = f"""# learning_rate: {learning_rate}\n# num_dense_layers: {num_dense_layers}\n# num_dense_nodes: {num_dense_nodes}\n# activation:{activation} \n# batch_size: 32\n# final loss: {error}\n# Training Time: {training_time}\n\n"""

    new_content = new_header + original_content

    with open(file_name, 'w') as file:
        file.write(new_content)

    results[ITERATION, :-2] = config #salva os parametros

    predicted_solution = np.empty((num_values, 1))
    for i in range(num_values):
        predicted_solution[i] = model.predict(input_data[i].reshape(1, -1))

    results[ITERATION, -2] = predicted_solution #salva as predicoes
    results[ITERATION, -1] = error #salva o erro

    if np.isnan(error):
        error = 10**5

    ITERATION += 1
    return error

ITERATION = 0

search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=1234,
)

sorted_results = results[results[:, -1].argsort()]

# Selecionar as 10 primeiras linhas após a ordenação.
top_10_results = sorted_results[:10]

for i, result in enumerate(top_10_results):
    label = f"LR: {result[0]}, Layers: {result[1]}, Nodes: {result[2]}, Activ.: {result[3]}"
    ax1.plot(y, result[-2].reshape(y.shape), linewidth=2, label=label)

ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_title('10 Melhores Configurações')
fig.savefig('grafico_comparacao.png', bbox_inches='tight')
plt.close(fig)

print(search_result.x)

fig_convergence = plot_convergence(search_result)
plt.savefig('convergence_plot.png')
plt.close(fig_convergence.figure)

fig_objective = plot_objective(search_result, show_points=True, size=3.8)
plt.savefig('objective_plot.png')
plt.close(fig_objective.figure)

# fig, ax = plt.subplots()
# ax = fig.add_subplot(111)

# nelx = 100  # Número de elementos em x
# nely = 100  # Número de elementos em y
# timesteps = 401

# x = np.linspace(0, 10, nelx + 1)
# y = np.linspace(0, 10, nely + 1)
# t = np.linspace(0, 2 * np.pi, timesteps)

# # Dados pra serem usados na predição
# test_x, test_y, test_t = np.meshgrid(x, y, t)
# test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y), np.ravel(test_t))).T

# # Predição da solução
# predicted_solution = pinn.model.predict(test_domain)
# predicted_solution = predicted_solution.reshape(
#     test_x.shape
# )

# animate_solution(
#     predicted_solution,
#     10,
#     10,
#     f"gaussian_pulse_rotation.mp4",
#     "Gaussian Pulse Rotation",
#     "u(x,y,t)",
#     t,
# )