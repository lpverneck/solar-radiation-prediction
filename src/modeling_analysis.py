"""Modeling Analysis

Processing of results generating tables and a refined data to be
analyzed on post processing.
"""


import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import hydroeval as he
from sklearn import metrics
import matplotlib.pyplot as plt

sns.set()


def vaf_metric(y_true, y_pred):
    return (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100


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
    stations = df["station"].unique()
    splitted_data = []
    for station in stations:
        station_data = df.loc[df["station"] == station]
        splitted_data.append(station_data)

    return splitted_data


def metrics_computation(data_list):
    """Calculates the metrics values based on the model output. (RMSE,
    MSE, MAE and R²)

    Parameters
    ----------
    data_list : list with the separated data corresponding to each
    station.

    Returns
    -------
    m_list : list with the separated data corresponding to each
    station with metrics included.
    """
    m_list = []
    for el in data_list:
        el["rmse"], el["mse"], el["mae"], el["r2"] = 0, 0, 0, 0
        el["vaf"], el["nse"] = 0, 0
        lines = el.shape[0]
        for i in range(lines):
            yt = np.asarray(el.iloc[i, 6], dtype=np.float32)
            yp = np.asarray(el.iloc[i, 7], dtype=np.float32)

            el.iloc[i, 10] = metrics.mean_squared_error(yt, yp, squared=False)
            el.iloc[i, 11] = metrics.mean_squared_error(yt, yp, squared=True)
            el.iloc[i, 12] = metrics.mean_absolute_error(yt, yp)
            el.iloc[i, 13] = metrics.r2_score(yt, yp)
            el.iloc[i, 14] = vaf_metric(yt, yp)
            el.iloc[i, 15] = he.nse(yt, yp)

        m_list.append(el)

    return m_list


def final_tab(data_list):
    """Performs the averages of each metrics evaluated in relation to
    the number of executions.

    Parameters
    ----------
    data_list : list with the separated data corresponding to each
    station containing the metrics values.

    Returns
    -------
    df : DataFrame containing the averages of each metric for each
    station.
    """
    f_list = []
    for el in data_list:
        d = {
            "station": el.iloc[0, 2],
            "rmse": el["rmse"].mean(),
            "(+-)rmse": el["rmse"].std(),
            "mse": el["mse"].mean(),
            "(+-)mse": el["mse"].std(),
            "mae": el["mae"].mean(),
            "(+-)mae": el["mae"].std(),
            "r2": el["r2"].mean(),
            "(+-)r2": el["r2"].std(),
            "vaf": el["vaf"].mean(),
            "(+-)vaf": el["vaf"].std(),
            "nse": el["nse"].mean(),
            "(+-)nse": el["nse"].std(),
        }

        f_list.append(d)

    return pd.DataFrame(f_list)


def select_by_execn(table, n_exec):
    """Selects the n_exec referenced data for all stations.

    Parameters
    ----------
    table : table with the final results divided.

    Returns
    -------
    json : json data file. containing 8x16 df.
    """
    li = []
    for item in table:
        li.append(item.loc[item["exec"] == n_exec])

    pd.concat(li).to_json("../results/exec_n_seletion.json")


# =============================================================================
# Analysis
# =============================================================================


raw = pd.read_json("../results/s1_results.json")

table_one = data_split(raw)

table_two = metrics_computation(table_one)

general_table = final_tab(table_two)

# df selected by exec number
select_by_execn(table_two, 15)
