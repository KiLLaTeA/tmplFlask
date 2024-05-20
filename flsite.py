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


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        try:
            X_new = np.array([[float(request.form['Pclass']),
                               float(request.form['Sex']),
                               float(request.form['Age']),
                               float(request.form['Siblings/Spouses Aboard']),
                               float(request.form['Parents/Children Aboard']),
                               float(request.form['Fare'])]])
            pred = results[loaded_model_knn.predict(X_new)[0]]
            return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model="Этот человек " + str(pred))
        except:
            return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, request='GET')


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        try:
            X_new = np.array([[float(request.form['Pclass']),
                               float(request.form['Sex']),
                               float(request.form['Age']),
                               float(request.form['Siblings/Spouses Aboard']),
                               float(request.form['Parents/Children Aboard']),
                               float(request.form['Fare'])]])
            pred = results[loaded_model_logistic.predict(X_new)[0]]
            return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model="Этот человек " + str(pred))
        except:
            return render_template('lab2.html', title="Логистическая регрессия", menu=menu, request='GET')


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
            dataset_x = pd.read_csv('model/titanic.csv')[
                ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
            ]
            dataset_y = pd.read_csv('model/titanic.csv')[['Survived']]
            labels = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
            critValues = []
            for i in labels:
                if i == 'Sex':
                    minValue = 0
                    maxValue = 1
                else:
                    minValue = dataset_x[i].min()
                    maxValue = dataset_x[i].max()
                critValues.append([minValue, maxValue])


            x_normal = []

            for i in range(len(labels)):
                tmp = np.array([critValues[i][0], float(request.form[labels[i]]), critValues[i][1]])
                tmp = pd.DataFrame(tmp)
                tmp_np = scaler.fit_transform(tmp)
                tmp_np = tmp_np.reshape(1, -1)
                x_normal.append(tmp_np[0][1])
            print(x_normal)

            pred = round(float(loaded_model_linear.predict([x_normal])[0]) * 100)
            if pred < 0:
                pred = '0 %'
            elif pred > 100:
                pred = '100 %'
            else:
                pred = str(pred) + ' %'

            return render_template('lab3.html', title="Линейная регрессия", menu=menu,
                                   class_model="Этот человек выжил с веротностью " + pred)


if __name__ == "__main__":
    app.run(debug=True)