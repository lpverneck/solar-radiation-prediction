import pandas as pd
import numpy as np
from sklearn import metrics


def data_split(df):
    stations = df['Station'].unique()
    splitted_data = []
    for station in stations:
        station_data = df.loc[df['Station'] == station]
        splitted_data.append(station_data)

    return splitted_data


def metrics_computation(y_true, y_pred):
    metrics_data = {
            'RMSE': metrics.mean_squared_error(y_true, y_pred, squared=False),
            'MAE': metrics.mean_absolute_error(y_true, y_pred),
            'MSE': metrics.mean_squared_error(y_true, y_pred, squared=True),
            'R2': metrics.r2_score(y_true, y_pred)
    }

    return metrics_data

# =============================================================================
# =============================================================================


df = pd.read_csv('results.csv')


LL = data_split(df)
