import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt

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

for key in ['axes.titlesize', 'axes.labelsize', 'legend.fontsize']:
    val = plt.rcParams.get(key)
    if isinstance(val, (int, float)):
        plt.rcParams[key] = val * _SCALE_FACTOR

_INKSCAPE_AVAILABLE = shutil.which('inkscape') is not None


def _pdf_to_eps(pdf_path: str, eps_path: str) -> bool:
    """Convert PDF to EPS using inkscape. Returns True on success."""
    try:
        subprocess.run(
            ['inkscape', pdf_path, f'--export-filename={eps_path}'],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class Plotter:
    def __init__(self, config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"

    def saveplot(self, name, xlabel=None, ylabel=None):
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        
        base_path = f"{self.config.name}/ode_plots/{name}"
        directory_path = Path(base_path).parent
        directory_path.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f"{base_path}.png")
        
        if _INKSCAPE_AVAILABLE:
            pdf_path = f"{base_path}.pdf"
            eps_path = f"{base_path}.eps"
            plt.savefig(pdf_path, format='pdf')
            if _pdf_to_eps(pdf_path, eps_path):
                os.remove(pdf_path)
            else:
                plt.savefig(eps_path, format='eps')
                os.remove(pdf_path)
        else:
            plt.savefig(f"{base_path}.eps", format='eps')
        
        plt.close()
