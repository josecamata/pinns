import deepxde as dde
import random
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
import skopt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective
import os
import csv

import sys
sys.path.append('../network')
from network import PINN

np.int = int

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
elif dde.backend.backend_name == "paddle":
    sin = dde.backend.paddle.sin
else:
    from deepxde.backend import tf
    sin = tf.sin

# Constante de difusividade
k = 10 ** -8

# Número de épocas
epochs = 10000 

# Semente para usar na busca dos parâmetros
seed = 9298745
dde.config.set_random_seed(seed)

# Tamanho dos lotes
batch_size = 32

# pde, top, right, down, left, ic
LOSS_WEIGHTS = [
    100, 
    0.1,
    0.1,
    0.1,
    0.1,
    10, 
]

# Definição da Rede Neural
def neural_network():
    
    # Função representado a PDE
    def pde(X, U):
        dU_x = dde.grad.jacobian(U, X, j=0)
        dU_y = dde.grad.jacobian(U, X, j=1)
        dU_t = dde.grad.jacobian(U, X, j=2)

        dU_xx = dde.grad.hessian(U, X, i=0, j=0)
        dU_yy = dde.grad.hessian(U, X, i=1, j=1)

        return dU_t + (5 - X[:, 1:2]) * dU_x + (X[:, 0:1] - 5) * dU_y - k * (dU_xx + dU_yy)
    
    # Função representando os valores de condição inicial da PDE
    def func_initial_condition(x):
        r = np.power(x[:, 0:1] - 5, 2) + np.power(x[:, 1:2] - 7.5, 2)
        return np.exp(-0.5 * r)
    
    # Monta o SOLVER usando como inicialização Glorot e Adam de otimizador
    pinn = PINN('Glorot uniform', 'adam')
    
    """Geometria:
        0 <= x <= 10, 
        0 <= y <= 10, 
        0 <= t <= 2 * pi
    """
    pinn.define_geometry(10, 10, 0, 2 * np.pi)
    pinn.define_pde(pde)
    
    # Todas as bordas inicializando com 0
    pinn.define_boundaries(0, 0, 0, 0)
    pinn.define_initial_condition(func_initial_condition)
    
    # 3000 amostras para treinamento
    pinn.training_data(3000)

    return pinn

pinn = neural_network()

""" Otimização dos Hiperparâmetros (HPO) """

# Número de chamadas para o HPO
n_calls = 40

# Taxas de Aprendizado usadas
dim_learning_rate = Categorical(categories=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2], name="learning_rate")

# Número de Camadas Ocultas
dim_num_dense_layers = Integer(low=3, high=8, name="num_dense_layers")

# Número de Neurônios em cada Camada Oculta
dim_num_dense_nodes = Categorical(categories=[20, 30, 40, 50, 60, 70, 80], name="num_dense_nodes")

# Funções de Ativação
dim_activation = Categorical(categories=["ReLU", "sigmoid", "tanh", "Swish", "sin"], name="activation")

dimensions = [
    dim_learning_rate,
    dim_num_dense_layers,
    dim_num_dense_nodes,
    dim_activation,
]

# Parametros padrão/inicial para começar a busca
default_parameters = [1e-3, 5, 60, "tanh"]

""" Parte para salvar a predição que será usada em um gráfico de recorte comparativo """
num_values = 128 # Malha idêntica ao do solver de elementos finitos
    
x = np.ones((num_values, 1)) * 5 # x = 5
y = np.linspace(0, 10, num_values).reshape(-1, 1) # y variável
t = np.ones((num_values, 1)) * (2 * np.pi) # t = 2 * pi

input_data = np.hstack((x, y, t))

# Array para armazenar dados ao longo das iterações
results = np.zeros((n_calls, 7), dtype=object)

# Função para buscar os hiperparâmetros
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_dense_nodes, activation):
    
    config = [learning_rate, num_dense_layers, num_dense_nodes, activation]
    global ITERATION

    print("\n # ITERAÇÃO: " + str(ITERATION))
    
    # Exbição dos HPs no cmd
    print("learning rate: {0:.1e}".format(learning_rate))
    print("num_dense_layers:", num_dense_layers)
    print("num_dense_nodes:", num_dense_nodes)
    print("activation:", activation)
    print()

    # Cria a Rede Neural com os parâmetros especificados
    model = pinn.create_model(config, LOSS_WEIGHTS)
    
    # Treino do Modelo
    error, training_time, best_step = pinn.train_model(model, epochs, batch_size, iteration_step = ITERATION)

    """ Parte para salvar no .dat os dados de hiperparâmetros usados naquele treinamento """
    
    file_name = f'outputs/loss/loss_{ITERATION}.dat'

    with open(file_name, 'r') as file:
        original_content = file.read()

    # Salva os dados usados de hiper parametros e outros dados, como loss final, tempo de treinamento e qual época foi obtido melhor modelo
    new_header = f"""# learning_rate: {learning_rate}\n# num_dense_layers: {num_dense_layers}\n# num_dense_nodes: {num_dense_nodes}\n# activation:{activation} \n# batch_size: 32\n# final loss: {error}\n# Training Time: {training_time}\n# Best Step: {best_step}\n\n"""

    new_content = new_header + original_content

    with open(file_name, 'w') as file:
        file.write(new_content)
    
    
    results[ITERATION, 0] = ITERATION #salva o ID para facilitar depois buscar a loss
    results[ITERATION, 1:5] = config # Salva os parâmetros
    
    """ Parte da Predição usando u(x = 5, y, 2*pi) """
    predicted_solution = np.empty((num_values, 1))
    for i in range(num_values):
        predicted_solution[i] = model.predict(input_data[i].reshape(1, -1))
    
    results[ITERATION, 5] = predicted_solution # Salva as predições
    results[ITERATION, 6] = error # Salva a loss obtida

    if np.isnan(error):
        error = 10**5 # Caso tenha algum problema
        
    ITERATION += 1
    return error

ITERATION = 0

# Método do skopt que fará de fato a otimização
search_result = gp_minimize(
    func=fitness,
    dimensions=dimensions,
    acq_func="EI",  # Expected Improvement.
    n_calls=n_calls,
    x0=default_parameters,
    random_state=seed,
)

""" Salvar isso em um CSV """

cols = ['ID', 'Taxa de Aprendizado', 'Número de Camadas Ocultas', 'Número de Neurônios por Camada', 'Ativação', 'Solução Predita', 'Loss']

file_name = 'results.csv'

# Salvando os dados no arquivo CSV
with open(file_name, mode='w', newline='') as arquivo_csv:
    escritor_csv = csv.writer(arquivo_csv)
    escritor_csv.writerow(cols)  # Header
    escritor_csv.writerows(results)  # Dados

# Print do melhor obtido depois de otimizar na função gp_minimize
print(search_result.x)

# Gráfico de convergência
fig_convergence = plot_convergence(search_result)
plt.savefig('plots/convergence_plot.png')
plt.close(fig_convergence.figure)

# Gráfico de comparação 2 a 2 dos hp's
fig_objective = plot_objective(search_result, show_points=True, size=3.8, cmap = 'RdYlBu_r')
plt.savefig('plots/objective_plot.png')
plt.close(fig_objective.figure)

# Arquivo final contendo informações mais gerais.
conteudo = f"""Treino com batch_size de {batch_size} e {epochs} épocas, além da seed ser {seed}.
Melhor configuração de parâmetros obtida: {search_result.x}
"""

with open('info.txt', 'w') as arquivo:
    arquivo.write(conteudo)