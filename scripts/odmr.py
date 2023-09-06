#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from scipy import linalg

base_path = os.getcwd() + "/../2022-09-22_hBN_Gd25/"

def get_path(n):
    return base_path + f"ODMR - Pulsed_{n}_spectrumdata.txt" 

def norm(ar):
    return ar / np.nanmax(ar)
    # return ar / ar[0]


def read_odmr_spectrum(path):
    """path to spectrumdata.txt file -> tau (in us), norm vals"""
    data = np.genfromtxt(path, skip_header=1, usecols=(0, 1, 2))
    START = 0
    END = None  # None equiv to end/-0
    # return data[START:END, 0] * 1e6, norm(data[START:END, 1] / data[START:END, 2])
    freqs = data[START:END, 0]
    sig = data[START:END, 1]
    ref = data[START:END, 2]
    # return tau * 1e6, (sig - ref) / (sig + ref)
    return freqs, sig / ref
    # return tau * 1e6, norm(sig - ref)


# ok... dry it off, add water, and try T1(t) -> 10ms exposures, less points?
config = {
    "film": {"path": get_path(22), "label": r"film", "color": "k"},
    "water": {"path": get_path(32), "label": r"water", "color": "b"},
    "gdsol": {"path": get_path(42), "label": r"Gd sol.", "color": "r"},
    # "gdsol2": {"path": get_path(51), "label": r"Gd sol. 2", "color": "g"},
    # "gdsol3": {"path": get_path(61), "label": r"Gd sol. 3", "color": "c"},
    # "gdsol4": {"path": get_path(71), "label": r"Gd sol. 4", "color": "m"},
}

for key in config.keys():
    config[key]["xs"], config[key]["ys"] = read_odmr_spectrum(config[key]["path"])


def model(params, x):
    a, tau, c = params
    return a * np.exp(-x / tau) + c


def resid(params, x, y):
    return y - model(params, x)

f1, ax1 = plt.subplots(figsize=(8, 6))
for key in config.keys():

    ax1.plot(
        config[key]["xs"],
        config[key]["ys"],
        "-o",
        ms=3,
        color=config[key]["color"],
        label=config[key]["label"],
    )

ax1.legend()
ax1.set_xlabel(r"Freq.")
ax1.set_ylabel("PL (a.u.)")
plt.show()
