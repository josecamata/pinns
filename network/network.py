import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import time
from deepxde.callbacks import ModelCheckpoint

class PINN:

    def __init__(self, INITIALIZER, OPTIMIZER):
        self.INITIALIZER = INITIALIZER
        self.OPTIMIZER = OPTIMIZER
    
    """
    Define a geometria espacial (Retângulo) e o intervalo de tempo para o problema.

    Argumentos:
    ----------
    WIDTH : float
        Largura do retângulo representando a geometria espacial.
    LENGTH : float
        Comprimento do retângulo representando a geometria espacial.
    T_START : float
        Tempo inicial do intervalo de tempo.
    T_END : float
        Tempo final do intervalo de tempo.
    """
    def define_geometry(self, WIDTH, LENGTH, T_START, T_END):
        
        self.WIDTH = WIDTH
        self.LENGTH = LENGTH
        self.T_START = T_START
        self.T_END = T_END

        geom = dde.geometry.Rectangle([0, 0], [self.WIDTH, self.LENGTH])
        time = dde.geometry.TimeDomain(self.T_START, self.T_END)
        self.geomtime = dde.geometry.GeometryXTime(geom, time)
    
    """
    Define a Equação Diferencial a ser resolvida. É importante a equação ser definida como um resíduo.

    Argumentos:
    ----------
    pde : function
        Função contendo os tensores para cálculo do resíduo da PDE.
    """
    def define_pde(self, pde):
        self.pde = pde
    
    """
    Define quando será considerado uma condição de contorno.
    No caso:
        f(x = 0, y, t)
        f(x = WIDTH, y, t)
        f(x, y = 0, t)
        f(x, y = LENGTH, t)
    """
    def _define_boundaries_functions(self):

        def boundary_top(X, on_boundary):
            _, y, _ = X
            return on_boundary and np.isclose(y, self.LENGTH)
        
        def boundary_right(X, on_boundary):
            x, _, _ = X
            return on_boundary and np.isclose(x, self.WIDTH)

        def boundary_down(X, on_boundary):
            _, y, _ = X
            return on_boundary and np.isclose(y, 0)
        
        def boundary_left(X, on_boundary):
            x, _, _ = X
            return on_boundary and np.isclose(x, 0)
        
        return [boundary_top, boundary_right, boundary_down, boundary_left]
    
    """
    Define os vetores que serão retornados para inicializar possíveis condições de contorno
    """
    def func_top(self, X):
        return np.ones(
            (len(X), 1)
        ) * self.U_DIR_TOP
    
    def func_right(self, X):
        return np.ones(
            (len(X), 1)
        ) * self.U_DIR_RIGHT
    
    def func_down(self, X):
        return np.ones(
            (len(X), 1)
        ) * self.U_DIR_DOWN
    
    def func_left(self, X):
        return np.ones(
            (len(X), 1)
        ) * self.U_DIR_LEFT
    
    """
    Define as condições de contorno (DIRICHLET) combinando os métodos ilustrados acima.

    Argumentos:
    ----------
    U_DIR_TOP : float
        Valor representando quanto a borda superior vale
    U_DIR_RIGHT : float
        Valor representando quanto a borda direita vale
    U_DIR_DOWN : float
        Valor representando quanto a borda inferior vale
    U_DIR_LEFT : float
        Valor representando quanto a borda esquerda vale
    """
    def define_boundaries(self, U_DIR_TOP, U_DIR_RIGHT, U_DIR_DOWN, U_DIR_LEFT):
        
        self.U_DIR_TOP = U_DIR_TOP
        self.U_DIR_RIGHT = U_DIR_RIGHT
        self.U_DIR_DOWN = U_DIR_DOWN
        self.U_DIR_LEFT = U_DIR_LEFT

        self.boundaries = self._define_boundaries_functions()

        self.bc_top = dde.DirichletBC(self.geomtime, self.func_top, self.boundaries[0])
        
        self.bc_right = dde.DirichletBC(self.geomtime, self.func_right, self.boundaries[1])
        
        self.bc_down = dde.DirichletBC(self.geomtime, self.func_down, self.boundaries[2])
        
        self.bc_left = dde.DirichletBC(self.geomtime, self.func_left, self.boundaries[3])
    
    """
    Define a condição inicial da PDE.
    Ou seja, quando f(x, y, t = 0)

    Argumentos:
    ----------
    func : function
        Função indicando qual serão os valores no interior do domínio quando o tempo é igual a 0.
    """
    def define_initial_condition(self, func):
        self.ic = dde.icbc.IC(self.geomtime, func, lambda _, on_initial: on_initial)
    
    """
    Monta de fato os dados de treinamento a partir das condições de contorno, condição inicial e resíduo da PDE especificados.

    Argumentos:
    ----------
    SAMPLE_POINTS : int
        Quantas amostras serão usadas para treinar a rede. 
    """
    def training_data(self, SAMPLE_POINTS):
        self.data = dde.data.TimePDE(
                self.geomtime,
                self.pde,
                [self.bc_top, self.bc_right, self.bc_down, self.bc_left, self.ic],
                num_domain=int(SAMPLE_POINTS),
                num_boundary=int(SAMPLE_POINTS / 4),
                num_initial=int(SAMPLE_POINTS / 2),
                num_test=int(SAMPLE_POINTS/2),
            )
    
    """
    Criação do modelo para resolver a PDE.

    Argumentos:
    ----------
    config : list
        Lista contendo hiperparâmetros como:
            - Learning Rate: Taxa de Aprendizado
            - Number Dense Layer: Número de Camadas Ocultas
            - Number Dense Nodes: Número de Neurônios em Cada Camada Oculta
            - Activation: Função de Ativação
    LOSS_WEIGHTS: list
        Lista contendo os bias/peso de cada componente da perda, a fim de auxiliar o modelo a priorizar certas características no treinamento.
            - 0 => Peso da PDE
            - 1 => Peso do contorno superior
            - 2 => Peso do contorno direito
            - 3 => Peso do contorno inferior
            - 4 => Peso do contorno esquerdo
            - 5 => Peso da condição inicial
    
    Retorno:
        Retorna o modelo criado.
    """
    def create_model(self, config, LOSS_WEIGHTS):

        learning_rate, num_dense_layers, num_dense_nodes, activation = config

        self.net = dde.maps.FNN(
            [3] + [num_dense_nodes] * num_dense_layers + [1],
            activation,
            self.INITIALIZER
        )

        model = dde.Model(self.data, self.net)
        model.compile(self.OPTIMIZER, lr=learning_rate, loss_weights=LOSS_WEIGHTS)
        
        return model
    
    """
    Treinamento do Modelo.

    Argumentos:
    ----------
    model : FNN
        Objeto do deepxde contendo o modelo da Rede Neural
    ITERATIONS: int
        Número de épocas que serão utilizadas para treino
    BATCH_SIZE: int
        Tamanho dos lotes para correção dos pesos pelo otimizador
    iteration_step: int
        Inteiro representando qual a iteração no treinamento dos hiperparâmetros para salvar corretamente os arquivos .dat
        
    Retorno:
        Retorna o erro e o tempo de treinamento. 
        Além disso, restaura o modelo que obteve melhor loss.
    ----------
    """
    def train_model(self, model, ITERATIONS, BATCH_SIZE, iteration_step):
        # CallBack para restaurar o Melhor Modelo
        checker = dde.callbacks.ModelCheckpoint(
            f"outputs/model/model_{iteration_step}.ckpt", save_better_only=True, period=1000, verbose=1
        )
        
        # Dispara um contador
        start_time = time.time()
        
        # Treina o modelo, obtendo um histórico das perdas 
        losshistory, train_state = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE, callbacks=[checker])
        
        # Encerra o contador
        end_time = time.time()
        
        # Obtém o tempo de treinamento
        training_time = end_time - start_time

        dde.saveplot(
            losshistory, train_state, issave=True, isplot=False,
            loss_fname= f"outputs/loss/loss_{iteration_step}.dat",
            train_fname = f"outputs/train/train{iteration_step}.dat",
            test_fname= f"outputs/test/test{iteration_step}.dat",
        )
        train = np.array(losshistory.loss_train).sum(axis=1).ravel()
        test = np.array(losshistory.loss_test).sum(axis=1).ravel()

        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        plt.savefig(f"outputs/loss_plot/loss_history_{iteration_step}")
        plt.close()

        error = test.min()
        
        # Restaura o melhor modelo
        if(train_state.best_step != 0):
            model.restore(f"outputs/model/model_{iteration_step}.ckpt-" + str(train_state.best_step) + ".pt", verbose=1)
        
        return train, test, error, training_time, train_state.best_step