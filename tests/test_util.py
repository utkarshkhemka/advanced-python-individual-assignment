import os

import ie_bike_model
from ie_bike_model.util import get_data_directory


def test_get_data_directory_gets_directory_next_to_modules():
    expected_data_directory = os.path.join(
        os.path.dirname(ie_bike_model.__file__), "data"
    )

    data_directory = get_data_directory()

    assert data_directory == expected_data_directory
