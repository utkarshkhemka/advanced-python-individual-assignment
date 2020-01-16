import os


def get_data_directory():
    """Gets directory where data is located.

    """
    lib_dir = os.path.dirname(__file__)
    data_dir = os.path.join(lib_dir, "data")
    return data_dir
