import os
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"

    def saveplot(self, name, xlabel=None, ylabel=None):
        """Save the current matplotlib figure with axis labels.

        If xlabel/ylabel are provided, they will be set on the current axes.
        If not provided (None), existing labels are preserved.
        """
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        full_path = f"{self.config.name}/ode_plots/{name}.png"
        directory_path = Path(full_path).parent
        directory_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(full_path)
        plt.close()
