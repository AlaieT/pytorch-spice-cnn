__all__ = ['plot_loss']

import matplotlib.pyplot as plt
import numpy as np
from typing import *

def plot_loss(dataset: np.ndarray, titles: List[str] = None, save_path: str = None):
    '''Plot graphs of provided data, created specialy for loss data...'''

    fig, axs = plt.subplots()

    for idx in range(dataset.shape[0]):
        axs.plot(np.linspace(start=0, stop=dataset.shape[1], num=dataset.shape[1]), dataset[idx])

    axs.legend(titles)
    axs.grid(visible=True)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()

    plt.close()
