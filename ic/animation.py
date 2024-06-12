import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import (
    FuncAnimation,
)
import numpy as np

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