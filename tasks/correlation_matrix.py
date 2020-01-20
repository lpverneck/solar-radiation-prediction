"""Correlation Matrix Plot

Displays a correlation matrix between the independent variables and the
dependent variable with coefficients, scatterplots and the distributions.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Correlation coef plot configuration
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    marker_size = abs(corr_r) * 10000
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
    df.columns = ['date', 'wind speed', 't max', 't min', 'humidity max',
                  'humidity min', 'vpd', 'evaporation', 'solar radiation']
    df = df.set_index('date')
    df = df[['solar radiation', 'wind speed', 't max', 't min', 'humidity max',
            'humidity min', 'vpd', 'evaporation']]

    # Plot
    sns.set(style='dark', font_scale=1.6, )
    g = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(corrdot)

    # Remove axis labels
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.82, fontsize=26)

    g.add_legend(title=station_name)
    g.savefig(station_name + ".png")
