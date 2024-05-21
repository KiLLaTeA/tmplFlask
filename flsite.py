import os
import json
import uuid
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from flask import Flask, render_template, url_for, jsonify, request

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


def read_data():
    if not os.path.exists('logs.json'):
        return []
    with open('logs.json', 'r') as f:
        data = json.load(f)
    return data


def write_data(logs):
    with open('logs.json', 'w') as f:
        json.dump(logs, f, indent=4)


@app.route("/")
def index():
    return render_template('index.html', title="Главная", menu=menu)


@app.route('/api/logs', methods=['GET'])
def get_records():
    data = read_data()
    return jsonify(data)


@app.route('/api/logs/<float:age_current>', methods=['GET'])
def get_record(age_current):
    data = read_data()
    record = next((item for item in data if item['Age'] == float(age_current)), None)
    if record:
        return jsonify(record)
    else:
        return jsonify({'error': 'Записи с таким возрастом не найдены!'}), 404


@app.route("/docs", methods=['GET'])
def docs():
    if request.method == 'GET':
        return render_template('docs.html', title="Документация", menu=menu, class_model='')


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

            data = read_data()
            form_data = request.form.to_dict()
            id_record = len(data) + 1
            record = {'id': id_record, **form_data}
            data.append(record)
            write_data(data)

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
            data = read_data()
            logs = {
                "id": str(uuid.uuid4()),
                "Pclass": float(request.form['Pclass']),
                "Sex": float(request.form['Sex']),
                "Age": float(request.form['Age']),
                "Siblings/Spouses Aboard": float(request.form['Siblings/Spouses Aboard']),
                "Parents/Children Aboard": float(request.form['Parents/Children Aboard']),
                "Fare": float(request.form['Fare']),
            }
            data.append(logs)
            write_data(data)
            return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model="Этот человек " + str(pred))
        except Exception as e:
            # Обработка ошибки и вывод её типа и сообщения
            print(f"Произошла ошибка: {type(e).__name__}")
            print(f"Сообщение об ошибке: {e}")
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


@app.route("/p_lab4", methods=['POST', 'GET'])
def f_lab4():
    if request.method == 'GET':
        return render_template('lab4.html', title="Дерево решений", menu=menu, class_model='')
    if request.method == 'POST':
        try:
            X_new = np.array([[float(request.form['Pclass']),
                               float(request.form['Sex']),
                               float(request.form['Age']),
                               float(request.form['Siblings/Spouses Aboard']),
                               float(request.form['Parents/Children Aboard']),
                               float(request.form['Fare'])]])
            pred = results[loaded_model_tree.predict(X_new)[0]]

            return render_template('lab4.html', title="Дерево решений", menu=menu,
                                   class_model="Этот человек " + str(pred))
        except:
            return render_template('lab4.html', title="Дерево решений", menu=menu, request='GET')


if __name__ == "__main__":
    app.run(debug=True)