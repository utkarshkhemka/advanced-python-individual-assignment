import os
import tempfile

import joblib
from sklearn.base import BaseEstimator

from ie_bike_model import model
from ie_bike_model.model import train_and_persist


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
