import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
sns.set()


# =============================================================================
# Results processing
# =============================================================================


def data_split(df):
    """Return a list with splitted data frames for each station.

    Parameters
    ----------
    df : data frame containing all results for each model along the
    independent executions.

    Returns
    -------
    splitted_data : list with the separated data corresponding to each
    station.
    """
    stations = df['station'].unique()
    splitted_data = []
    for station in stations:
        station_data = df.loc[df['station'] == station]
        splitted_data.append(station_data)

    return splitted_data


def metrics_computation(data_list):
    m_list = []
    for el in data_list:
        el['rmse'], el['mse'], el['mae'], el['r2'] = 0, 0, 0, 0
        lines = el.shape[0]
        for i in range(lines):
            yt = np.asarray(el.iloc[i, 6], dtype=np.float32)
            yp = np.asarray(el.iloc[i, 7], dtype=np.float32)

            el.iloc[i, 10] = metrics.mean_squared_error(yt, yp, squared=False)
            el.iloc[i, 11] = metrics.mean_squared_error(yt, yp, squared=True)
            el.iloc[i, 12] = metrics.mean_absolute_error(yt, yp)
            el.iloc[i, 13] = metrics.r2_score(yt, yp)

        m_list.append(el)

    return m_list


def final_tab(data_list):
    f_list = []
    for el in data_list:
        n = el.shape[0]
        d = {
             'station': el.iloc[0, 2],
             'rmse': (el['rmse'].sum()) / n,
             'mse': (el['mse'].sum()) / n,
             'mae': (el['mae'].sum()) / n,
             'r2': (el['r2'].sum()) / n
        }

        f_list.append(d)

    return pd.DataFrame(f_list)


# =============================================================================
# Plot
# =============================================================================


def plot_v1(data):
    # Create the figure and axis objects I'll be plotting on
    fig, ax = plt.subplots()
    
    # Plot the bars
    ax.bar(np.arange(len(data)), data, align='center')
    
    # Show the 50% mark, which would indicate an equal
    # number of tasks being completed by the robot and the
    # human. There are 39 tasks total, so 50% is 19.5
    ax.hlines(19.5, -0.5, 5.5, linestyle='--', linewidth=1)
    
    # Set a reasonable y-axis limit
    ax.set_ylim(0, 40)
    
    # Apply labels to the bars so you know which is which
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(["\n".join(x) for x in data.index])
    
    return fig, ax


def plot_v2(data):
    # Create the bar plot
    ax = sns.barplot(
        x="robot", y="robot_tasks", hue="inference",
        order=["fixed", "reactive", "predictive"],
        hue_order=["oracle", "bayesian"],
        data=data)

    # Plot the 50% line
    ax.hlines(19.5, -0.5, 4.5, linestyle='--', linewidth=1)
    ax.set_ylim(0, 40)
    
    # Return the figure object and axis
    return plt.gcf(), ax


def plot_v3(data):
    # Specify that I want each subplot to correspond to
    # a different robot type
    g = sns.FacetGrid(
        data,
        col="robot",
        col_order=["fixed", "reactive", "predictive"],
        sharex=False)

    # Create the bar plot on each subplot
    g.map(
        sns.barplot,
        "robot", "robot_tasks", "inference",
        hue_order=["oracle", "bayesian"])

    # Now I need to draw the 50% lines on each subplot
    # separately
    axes = np.array(g.axes.flat)
    for ax in axes:
        ax.hlines(19.5, -0.5, 0.5, linestyle='--', linewidth=1)
        ax.set_ylim(0, 40)

    # Return the figure and axes objects
    return plt.gcf(), axes


def set_labels(fig, axes):
    # These are the labels of each subplot
    labels = ["Fixed", "Reactive", "Predictive"]
    
    # Iterate over each subplot and set the labels
    for i, ax in enumerate(axes):

        # Set the x-axis ticklabels
        ax.set_xticks([-.2, .2])
        ax.set_xticklabels(["Oracle", "Bayesian"])

        # Set the label for each subplot
        ax.set_xlabel(labels[i])
        
        # Remove the y-axis label and title
        ax.set_ylabel("")
        ax.set_title("")
    
    # Set the y-axis label only for the left subplot
    axes.flat[0].set_ylabel("Number of tasks")
    
    # Remove the "spines" (the lines surrounding the subplot)
    # including the left spine for the 2nd and 3rd subplots
    sns.despine(ax=axes[1], left=True)
    sns.despine(ax=axes[2], left=True)

    # Set the overall title for the plot
    fig.suptitle("Single-agent tasks completed by the robot", fontsize=12, x=0.55)


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def get_colors():
    return np.array([
        [0.1, 0.1, 0.1],          # black
        [0.4, 0.4, 0.4],          # very dark gray
        [0.7, 0.7, 0.7],          # dark gray
        [0.9, 0.9, 0.9],          # light gray
        [0.984375, 0.7265625, 0], # dark yellow
        [1, 1, 0.9]               # light yellow
    ])

# sns.palplot(get_colors())


def color_bars(axes, colors):
    # Iterate over each subplot
    for i in range(3):

        # Pull out the dark and light colors for
        # the current subplot
        dark_color = colors[i*2]
        light_color = colors[i*2 + 1]

        # These are the patches (matplotlib's terminology
        # for the rectangles corresponding to the bars)
        p1, p2 = axes[i].patches

        # The first bar gets the dark color
        p1.set_color(dark_color)
        
        # The second bar gets the light color, plus
        # hatch marks int he dark color
        p2.set_color(light_color)
        p2.set_edgecolor(dark_color)
        p2.set_hatch('////')


def set_size(fig):
    fig.set_size_inches(6, 3)
    plt.tight_layout()



# =============================================================================
# =============================================================================


df = pd.read_json('results.json')


LL = data_split(df)


TT = metrics_computation(LL)


FF = final_tab(TT)
