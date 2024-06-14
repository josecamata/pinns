import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

class PINN:

    def __init__(self, INITIALIZER, OPTIMIZER):
        self.INITIALIZER = INITIALIZER
        self.OPTIMIZER = OPTIMIZER

    def define_geometry(self, WIDTH, LENGTH, T_START, T_END):
        self.WIDTH = WIDTH
        self.LENGTH = LENGTH
        self.T_START = T_START
        self.T_END = T_END

        geom = dde.geometry.Rectangle([0, 0], [self.WIDTH, self.LENGTH])
        time = dde.geometry.TimeDomain(self.T_START, self.T_END)
        self.geomtime = dde.geometry.GeometryXTime(geom, time)
        
    def define_pde(self, pde):
        self.pde = pde

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

    def define_initial_condition(self, func):
        self.ic = dde.icbc.IC(self.geomtime, func, lambda _, on_initial: on_initial)
        
    def training_data(self, SAMPLE_POINTS):
        self.data = dde.data.TimePDE(
                self.geomtime,
                self.pde,
                [self.bc_top, self.bc_right, self.bc_down, self.bc_left, self.ic],
                num_domain=int(SAMPLE_POINTS),
                num_boundary=int(SAMPLE_POINTS / 4),
                num_initial=int(SAMPLE_POINTS / 2),
            )

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
        # losshistory, trainstate = self.model.train(
        #     iterations=ITERATIONS,
        #     batch_size=BATCH_SIZE,
        # )

        # dde.saveplot(losshistory, trainstate, issave=True, isplot=True)
        # plt.show()
        # plt.savefig(title)
        # plt.close()

    def train_model(self, model, ITERATIONS, BATCH_SIZE, iteration_step):
        
        losshistory, train_state = model.train(iterations=ITERATIONS, batch_size=BATCH_SIZE)
        dde.saveplot(
            losshistory, train_state, issave=True, isplot=False,
            loss_fname= f"/content/outputs/loss/loss_{iteration_step}.dat",
            train_fname = f"/content/outputs/train/train{iteration_step}.dat",
            test_fname= f"/content/outputs/test/test{iteration_step}.dat",
        )
        train = np.array(losshistory.loss_train).sum(axis=1).ravel()
        test = np.array(losshistory.loss_test).sum(axis=1).ravel()

        error = test.min()
        return error


