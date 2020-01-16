import datetime as dt
import os
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from scipy.stats import skew
from xgboost import XGBRegressor

from ie_bike_model.util import read_data, get_season, get_model_path


US_HOLIDAYS = calendar().holidays()


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

    return hour


def dummify(hour, known_columns=None):
    hour = pd.get_dummies(hour)
    if known_columns is not None:
        for col in known_columns:
            if col not in hour.columns:
                hour[col] = 0

        hour = hour[known_columns]

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


def postprocess(hour):
    # Avoid modifying the original dataset at the cost of RAM
    hour = hour.copy()

    hour.columns = hour.columns.str.replace("[\[\]\<]", "_")
    return hour


def train_and_persist(model_dir=None, hour_path=None):
    hour = read_data(hour_path)
    hour = preprocess(hour)
    hour = dummify(hour)
    hour = postprocess(hour)

    # TODO: Implement other models?
    model = train_xgboost(hour)

    model_path = get_model_path(model_dir)

    joblib.dump(model, model_path)


def get_input_dict(parameters):
    hour_original = read_data()
    base_year = pd.to_datetime(hour_original["dteday"]).min().year

    date = parameters["date"]

    is_holiday = date in US_HOLIDAYS
    is_weekend = date.weekday() in (5, 6)

    row = pd.Series(
        {
            "dteday": date.strftime("%Y-%m-%d"),
            "season": get_season(date),
            "yr": date.year - base_year,
            "mnth": date.month,
            "hr": date.hour,
            "holiday": 1 if is_holiday else 0,
            "weekday": (date.weekday() + 1) % 7,
            "workingday": 0 if is_holiday or is_weekend else 1,
            "weathersit": parameters["weathersit"],
            "temp": parameters["temperature_C"] / 41.0,
            "atemp": parameters["feeling_temperature_C"] / 50.0,
            "hum": parameters["humidity"] / 100.0,
            "windspeed": parameters["windspeed"] / 67.0,
            "cnt": 1,  # Dummy, unused for prediction
        }
    )

    dummified_original = dummify(preprocess(hour_original))

    df = pd.DataFrame([row])
    df = preprocess(df)
    df = dummify(df, dummified_original.columns)
    df = postprocess(df)

    df = df.drop(columns=["dteday", "atemp", "casual", "registered", "cnt"])

    assert len(df) == 1

    return df.iloc[0].to_dict()


def predict(parameters, model_dir=None):
    """Returns model prediction.

    """
    model_path = get_model_path(model_dir)
    if not os.path.exists(model_path):
        train_and_persist(model_dir)

    model = joblib.load(model_path)

    input_dict = get_input_dict(parameters)
    X_input = pd.DataFrame([pd.Series(input_dict)])

    result = model.predict(X_input)

    # Undo np.sqrt(hour["cnt"])
    return int(result ** 2)
