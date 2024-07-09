import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import (
    FuncAnimation,
)
import numpy as np

# Função para animar a solução
def animate_solution(data, WIDTH, LENGTH, filename, title, label, t_data):
    fig, ax = plt.subplots(figsize=(7, 7))

    # Cria a imagem inicial com a barra de cores
    im = ax.imshow(
        data[:, :, 0],
        origin="lower",
        cmap="jet",
        interpolation="bilinear",
        extent=[0, WIDTH, 0, LENGTH],
    )
    cb = plt.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Atualiza os frames
    def updatefig(k):
        # Atualiza a imagem em si
        im.set_array(data[:, :, k])
        im.set_clim(
            vmin=data[:, :, k].min(), vmax=data[:, :, k].max()
        )  # Atualiza também as cores limitantes

        cb.update_normal(im)

        ax.set_title(f"{title}, t = {t_data[k]:.2f}")

        return [im]

    ani = animation.FuncAnimation(
        fig, updatefig, frames=range(data.shape[2]), interval=50, blit=True
    )
    ani.save(filename, writer="ffmpeg")
    
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