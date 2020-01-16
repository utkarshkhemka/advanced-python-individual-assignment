import datetime as dt
import os

import pandas as pd

WINTER_SOLSTICE = dt.datetime(2000, 12, 21)
SPRING_EQUINOX = dt.datetime(2000, 3, 21)
SUMMER_SOLSTICE = dt.datetime(2000, 6, 21)
FALL_EQUINOX = dt.datetime(2000, 9, 21)


def get_data_directory():
    """Gets directory where data is located.

    """
    lib_dir = os.path.dirname(__file__)
    data_dir = os.path.join(lib_dir, "data")
    return data_dir


def get_model_path(model_dir=None):
    if model_dir is None:
        model_dir = os.path.dirname(__file__)

    model_path = os.path.join(model_dir, "model.pkl")
    return model_path


def read_data(hour_path=None):
    if hour_path is None:
        hour_path = os.path.join(get_data_directory(), "hour.csv")

    hour = pd.read_csv(hour_path, index_col="instant", parse_dates=True)
    return hour


def get_season(date):
    """Get season, assuming fixed equinoxes and solstices.

    """
    # For comparison purposes
    date = date.replace(year=2000)

    # Use correct encoding for seasons!
    if SPRING_EQUINOX <= date < SUMMER_SOLSTICE:
        return 2
    elif SUMMER_SOLSTICE <= date < FALL_EQUINOX:
        return 3
    elif FALL_EQUINOX <= date < WINTER_SOLSTICE:
        return 4
    else:
        return 1
