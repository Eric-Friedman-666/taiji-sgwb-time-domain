import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import h5py
from copy import copy
import pandas as pd
from tqdm import tqdm
from py_code.parameters import *
from py_code.model import *
from multiprocessing import Pool, cpu_count
import corner
import os
from getdist import MCSamples, plots

# Render all Matplotlib axis labels and legends through LaTeX.
plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})
