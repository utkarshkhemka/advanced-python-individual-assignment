import os

import pandas as pd


def get_data_directory():
    """Gets directory where data is located.

    """
    lib_dir = os.path.dirname(__file__)
    data_dir = os.path.join(lib_dir, "data")
    return data_dir


def read_data(hour_path=None):
    if hour_path is None:
        hour_path = os.path.join(get_data_directory(), "hour.csv")

    hour = pd.read_csv(hour_path, index_col="instant", parse_dates=True)
    return hour
