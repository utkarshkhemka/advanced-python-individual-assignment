import datetime as dt
from flask import Flask, request

from ie_bike_model.model import predict

app = Flask(__name__)


@app.route("/")
def hello():
    name = request.args.get("name", "World")
    return "Hello, " + name + "!"


@app.route("/predict", methods=["GET"])
def get_predict():
    parameters = dict(request.args)
    parameters["date"] = dt.datetime.fromisoformat(parameters["date"])
    parameters["weathersit"] = int(parameters["weathersit"])
    parameters["temperature_C"] = float(parameters["temperature_C"])
    parameters["feeling_temperature_C"] = float(parameters["feeling_temperature_C"])
    parameters["humidity"] = float(parameters["humidity"])
    parameters["windspeed"] = float(parameters["windspeed"])

    model = str(request.args.get("model", "xgboost"))
    result = predict(parameters, model_name=model)
    return {"No. of bikes": result}
