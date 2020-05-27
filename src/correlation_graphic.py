"""Correlation Graphic

Displays a correlation matrix graphically between the variables with
coefficients, scatterplots and the distributions.
"""


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt


t = time.time()


def time_display(t, flag=""):
    """Generates a display that shows the elapsed time of computation.

    Parameters
    ----------
    t : initial time in seconds since the Epoch.

    flag : {'mid', 'end'}
        Flag to chosse the position in the code.
        - 'mid' chooses the display in the middle of the computation.
        - 'end' chooses the display in the end of the computation.
    """
    elapsed = time.time() - t
    if (flag == "end"):
        print("="*45 + "\nTotal Elapsed Time:\n")
        print(int(elapsed / 3600), 'hours', int((elapsed % 3600) / 60),
              'minutes', int((elapsed % 3600) % 60), 'seconds')
        print("="*45 + "\n")
        print(datetime.now())
    elif (flag == "mid"):
        print(' - [', int(elapsed / 3600), 'h', int((elapsed % 3600) / 60),
              'm', int((elapsed % 3600) % 60), 's]', end='')


# Correlation coefficient plot configuration
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    marker_size = (abs(corr_r) * 10000) + 1500
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 15
    ax.annotate(corr_text, [.5, .5],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)


# Selecting data folder
data_dir = os.getcwd()[:-6] + "\\data\\raw"
files = os.listdir(data_dir)
files.pop(0)


# Load data from each station
for station in files:
    station_name = station[:-5]
    df = pd.read_excel(data_dir + "\\" + station, header=1)

    # Data preprocessing
    df.columns = ['Data', 'WS', 'T_max', 'T_min', 'H_max', 'H_min', 'DPV',
                  'Eo', 'RS']
    df = df.set_index('Data')
    df = df[['RS', 'WS', 'T_max', 'T_min', 'H_max', 'H_min', 'DPV', 'Eo']]

    # Plot
    sns.set(style='dark', font_scale=1.6, font='serif')
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"],
        "axes.linewidth": 2,
        "axes.edgecolor": "black"
    },)
    g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, line_kws={'color': 'red'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'}, color='aqua',
               hist_kws=dict(edgecolor="k", linewidth=0.3))
    # g.map_diag(sns.rugplot, color='gray')
    g.map_upper(corrdot)

    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)

    g.savefig(station_name + ".png")
    g.savefig(station_name + ".pdf")


# Display the elapsed time
time_display(t, flag="end")
