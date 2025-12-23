import os
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

# Increase text sizes by 25% for titles, axis labels, and legend
_DEFAULT_FONT_SIZE = plt.rcParams.get('font.size', 10)
_SCALE_FACTOR = 1.25

plt.rcParams.update({
    'font.size': _DEFAULT_FONT_SIZE * _SCALE_FACTOR,
    'axes.titlesize': plt.rcParams.get('axes.titlesize', 'large'),
    'axes.labelsize': plt.rcParams.get('axes.labelsize', 'medium'),
    'legend.fontsize': plt.rcParams.get('legend.fontsize', 'medium'),
    'xtick.labelsize': plt.rcParams.get('xtick.labelsize', 'medium'),
    'ytick.labelsize': plt.rcParams.get('ytick.labelsize', 'medium'),
})

# Scale specific sizes if they are numeric
for key in ['axes.titlesize', 'axes.labelsize', 'legend.fontsize']:
    val = plt.rcParams.get(key)
    if isinstance(val, (int, float)):
        plt.rcParams[key] = val * _SCALE_FACTOR


class Plotter:
    def __init__(self, config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"

    def saveplot(self, name, xlabel=None, ylabel=None):
        """Save the current matplotlib figure with axis labels.

        If xlabel/ylabel are provided, they will be set on the current axes.
        If not provided (None), existing labels are preserved.
        Saves both PNG and EPS formats.
        """
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        
        base_path = f"{self.config.name}/ode_plots/{name}"
        directory_path = Path(base_path).parent
        directory_path.mkdir(parents=True, exist_ok=True)
        
        # Save PNG
        plt.savefig(f"{base_path}.png")
        # Save EPS
        plt.savefig(f"{base_path}.eps", format='eps')
        
        plt.close()
