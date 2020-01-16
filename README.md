# Bike sharing prediction model

## Usage

To install the library:

```
$ # pip install ie_bike_model  # If I ever upload this to PyPI, which I won't
$ pip install .
```

Basic usage:

```python
>>> import datetime as dt
>>> from ie_bike_model.model import train_and_persist, predict
>>> train_and_persist()
>>> predict({
...     "date": dt.datetime(2011, 1, 1, 0, 0, 0),
...     "weathersit": 1,
...     "temperature_C": 9.84,
...     "feeling_temperature_C": 14.395,
...     "humidity": 81.0,
...     "windspeed": 0.0,
... })
1
```

## Development

To install a development version of the library:

```
$ flit install --symlink --extras=dev
```

To run the tests:

```
$ pytest
```

To measure the coverage:

```
$ pytest --cov=ie_bike_model
```

## Trivia

Total implementation time: **4 hours 30 minutes** 🏁
