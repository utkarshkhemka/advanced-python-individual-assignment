import os
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from xgboost import XGBRegressor

from ie_bike_model.util import get_data_directory


def feature_engineering(hour):
    # Avoid modifying the original dataset at the cost of RAM
    hour = hour.copy()

    # Rented during office hours
    hour["IsOfficeHour"] = np.where(
        (hour["hr2"] >= 9) & (hour["hr2"] < 17) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsOfficeHour"] = hour["IsOfficeHour"].astype("category")

    # Rented during daytime
    hour["IsDaytime"] = np.where((hour["hr2"] >= 6) & (hour["hr2"] < 22), 1, 0)
    hour["IsDaytime"] = hour["IsDaytime"].astype("category")

    # Rented during morning rush hour
    hour["IsRushHourMorning"] = np.where(
        (hour["hr2"] >= 6) & (hour["hr2"] < 10) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourMorning"] = hour["IsRushHourMorning"].astype("category")

    # Rented during evening rush hour
    hour["IsRushHourEvening"] = np.where(
        (hour["hr2"] >= 15) & (hour["hr2"] < 19) & (hour["weekday2"] == 1), 1, 0
    )
    hour["IsRushHourEvening"] = hour["IsRushHourEvening"].astype("category")

    # Rented during most busy season
    hour["IsHighSeason"] = np.where((hour["season2"] == 3), 1, 0)
    hour["IsHighSeason"] = hour["IsHighSeason"].astype("category")

    # binning temp, atemp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    hour["temp_binned"] = pd.cut(hour["temp2"], bins).astype("category")
    hour["hum_binned"] = pd.cut(hour["hum2"], bins).astype("category")

    return hour


def preprocess(hour):
    # Avoid modifying the original dataset at the cost of RAM
    hour = hour.copy()

    # creating duplicate columns for feature engineering
    hour["hr2"] = hour["hr"]
    hour["season2"] = hour["season"]
    hour["temp2"] = hour["temp"]
    hour["hum2"] = hour["hum"]
    hour["weekday2"] = hour["weekday"]

    # Change dteday to date time
    hour["dteday"] = pd.to_datetime(hour["dteday"])

    # Convert the data type to eithwe category or to float
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    ]
    for col in int_hour:
        hour[col] = hour[col].astype("category")

    # Log of the count since log0 is invalid so we use log1p.
    logws = round(skew(np.log1p(hour.windspeed)), 4)
    # Sqrt of the count
    sqrtws = round(skew(np.sqrt(hour.windspeed)), 4)
    hour["windspeed"] = np.log1p(hour.windspeed)

    # Log of the count
    logcnt = round(skew(np.log(hour.cnt)), 4)
    # Sqrt of the count
    sqrtcnt = round(skew(np.sqrt(hour.cnt)), 4)
    hour["cnt"] = np.sqrt(hour.cnt)

    hour = feature_engineering(hour)

    # dropping duplicated rows used for feature engineering
    hour = hour.drop(columns=["hr2", "season2", "temp2", "hum2", "weekday2"])

    hour = pd.get_dummies(hour)

    return hour


def split_train_test(hour):
    # Avoid modifying the original dataset at the cost of RAM
    hour = hour.copy()

    # Split into train and test
    hour_train, hour_test = hour.iloc[0:15211], hour.iloc[15212:17379]
    train = hour_train.drop(columns=["dteday", "casual", "atemp", "registered"])
    test = hour_test.drop(columns=["dteday", "casual", "registered", "atemp"])

    # seperate the independent and target variable on testing data
    train_X = train.drop(columns=["cnt"], axis=1)
    train_y = train["cnt"]

    # seperate the independent and target variable on test data
    test_X = test.drop(columns=["cnt"], axis=1)
    test_y = test["cnt"]

    return train_X, test_X, train_y, test_y


def train_xgboost(hour):
    # Avoid modifying the original dataset at the cost of RAM
    hour = hour.copy()

    hour_d = pd.get_dummies(hour)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    hour_d.columns = [
        regex.sub("_", col) if any(x in str(col) for x in set(("[", "]", "<"))) else col
        for col in hour_d.columns.values
    ]

    hour_d = hour_d.select_dtypes(exclude="category")

    hour_d_train_x, _, hour_d_train_y, _, = split_train_test(hour_d)

    xgb = XGBRegressor(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=15,
        objective="reg:squarederror",
        subsample=0.8,
        colsample_bytree=1,
        seed=1234,
        gamma=1,
    )

    xgb.fit(hour_d_train_x, hour_d_train_y)
    return xgb


def train_and_persist(model_dir=None, hour_path=None):
    if model_dir is None:
        model_dir = os.path.dirname(__file__)

    if hour_path is None:
        hour_path = os.path.join(get_data_directory(), "hour.csv")

    hour = pd.read_csv(hour_path, index_col="instant", parse_dates=True)
    hour = preprocess(hour)

    # TODO: Implement other models?
    model = train_xgboost(hour)

    model_path = os.path.join(model_dir, "model.pkl")

    joblib.dump(model, model_path)
