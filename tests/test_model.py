import datetime as dt
import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest

import joblib
from sklearn.base import BaseEstimator

from ie_bike_model import model
from ie_bike_model.model import train_and_persist, get_input_dict, predict


def test_train_and_persist_persists_model():
    with tempfile.TemporaryDirectory() as tmp_dir:
        expected_model_path = os.path.join(tmp_dir, "model.pkl")
        train_and_persist(model_dir=tmp_dir)

        assert os.path.exists(expected_model_path)
        clf = joblib.load(expected_model_path)

        assert isinstance(clf, BaseEstimator)


def test_train_and_persist_persists_model_with_no_parameters():
    expected_model_path = os.path.join(os.path.dirname(model.__file__), "model.pkl")

    train_and_persist()

    assert os.path.exists(expected_model_path)
    clf = joblib.load(expected_model_path)

    assert isinstance(clf, BaseEstimator)


def test_get_input_dict_returns_expected_result():
    expected_result = {
        "temp": 0.44,
        "hum": 0.77,
        "windspeed": 0.08581065641311086,
        "season_1": 0.0,
        "season_2": 0.0,
        "season_3": 0.0,
        "season_4": 1.0,
        "yr_0": 0.0,
        "yr_1": 1.0,
        "mnth_1": 0.0,
        "mnth_2": 0.0,
        "mnth_3": 0.0,
        "mnth_4": 0.0,
        "mnth_5": 0.0,
        "mnth_6": 0.0,
        "mnth_7": 0.0,
        "mnth_8": 0.0,
        "mnth_9": 0.0,
        "mnth_10": 1.0,
        "mnth_11": 0.0,
        "mnth_12": 0.0,
        "hr_0": 0.0,
        "hr_1": 1.0,
        "hr_2": 0.0,
        "hr_3": 0.0,
        "hr_4": 0.0,
        "hr_5": 0.0,
        "hr_6": 0.0,
        "hr_7": 0.0,
        "hr_8": 0.0,
        "hr_9": 0.0,
        "hr_10": 0.0,
        "hr_11": 0.0,
        "hr_12": 0.0,
        "hr_13": 0.0,
        "hr_14": 0.0,
        "hr_15": 0.0,
        "hr_16": 0.0,
        "hr_17": 0.0,
        "hr_18": 0.0,
        "hr_19": 0.0,
        "hr_20": 0.0,
        "hr_21": 0.0,
        "hr_22": 0.0,
        "hr_23": 0.0,
        "holiday_0": 1.0,
        "holiday_1": 0.0,
        "weekday_0": 0.0,
        "weekday_1": 1.0,
        "weekday_2": 0.0,
        "weekday_3": 0.0,
        "weekday_4": 0.0,
        "weekday_5": 0.0,
        "weekday_6": 0.0,
        "workingday_0": 0.0,
        "workingday_1": 1.0,
        "weathersit_1": 1.0,
        "weathersit_2": 0.0,
        "weathersit_3": 0.0,
        "weathersit_4": 0.0,
        "IsOfficeHour_0": 1.0,
        "IsOfficeHour_1": 0.0,
        "IsDaytime_0": 1.0,
        "IsDaytime_1": 0.0,
        "IsRushHourMorning_0": 1.0,
        "IsRushHourMorning_1": 0.0,
        "IsRushHourEvening_0": 1.0,
        "IsRushHourEvening_1": 0.0,
        "IsHighSeason_0": 1.0,
        "IsHighSeason_1": 0.0,
        "temp_binned_(0.0, 0.19_": 0.0,
        "temp_binned_(0.19, 0.49_": 1.0,
        "temp_binned_(0.49, 0.69_": 0.0,
        "temp_binned_(0.69, 0.89_": 0.0,
        "temp_binned_(0.89, 1.0_": 0.0,
        "hum_binned_(0.0, 0.19_": 0.0,
        "hum_binned_(0.19, 0.49_": 0.0,
        "hum_binned_(0.49, 0.69_": 0.0,
        "hum_binned_(0.69, 0.89_": 1.0,
        "hum_binned_(0.89, 1.0_": 0.0,
    }

    parameters = {
        "date": dt.datetime(2012, 10, 1, 1, 0, 0),
        "weathersit": 1,
        "temperature_C": 18.04,
        "feeling_temperature_C": 21.97,
        "humidity": 77.0,
        "windspeed": 6.0032,
    }

    result = get_input_dict(parameters)

    assert result == expected_result


def test_predict_returns_expected_output():
    parameters = {
        "date": dt.datetime(2011, 1, 1, 0, 0, 0),
        "weathersit": 1,
        "temperature_C": 9.84,
        "feeling_temperature_C": 14.395,
        "humidity": 81.0,
        "windspeed": 0.0,
    }

    # int(np.round(xgb.predict(hour_d_test_x.iloc[0:1]) ** 2))
    expected_result = 1

    result = predict(parameters)

    assert result == expected_result
