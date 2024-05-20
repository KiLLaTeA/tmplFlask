import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from flask import Flask, render_template, url_for, request

scaler = preprocessing.MinMaxScaler()

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"},
        {"name": "Лаба 4", "url": "p_lab4"}]

loaded_model_knn = pickle.load(open('model/titanic_knn', 'rb'))
loaded_model_logistic = pickle.load(open('model/Titanic_pickle_file', 'rb'))
loaded_model_linear = pickle.load(open('model/titanic_linear', 'rb'))
loaded_model_tree = pickle.load(open('model/titanic_tree', 'rb'))

results = ['не выжил', 'выжил']


@app.route("/")
def index():
    return render_template('index.html', title="Главная", menu=menu)


if __name__ == "__main__":
    app.run(debug=True)