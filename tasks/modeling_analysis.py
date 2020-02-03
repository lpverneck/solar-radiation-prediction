import pandas as pd
import numpy as np
from sklearn import metrics


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


def metrics_computation(df):
    for i in range(len(df)):
        metrics.mean_squared_error(df.iloc[0, 6], df.iloc[0, 7], squared=False)
    df['rmse'] = metrics.mean_squared_error(df['y_true'], df['y_pred'], squared=False)
    
    # 'MAE': metrics.mean_absolute_error(y_true, y_pred),
    # 'MSE': metrics.mean_squared_error(y_true, y_pred, squared=True),
    # 'R2': metrics.r2_score(y_true, y_pred)

    return metrics_data

# =============================================================================
# =============================================================================


dff = pd.read_csv('results.csv')


LL = data_split(df)
