import os
from pathlib import Path


def train_and_persist(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")

    Path(model_path).touch()
