import pickle as pk
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import column_or_1d
from sklearn.linear_model import LinearRegression

dataset_x = pd.read_csv('titanic.csv')[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
dataset_y = pd.read_csv('titanic.csv')[['Survived']]

data_sex = pd.DataFrame(dataset_x['Sex'])
data_sex = column_or_1d(data_sex)

labelencoder = LabelEncoder()
data_sex = labelencoder.fit_transform(data_sex)

dataset_x['Sex'] = data_sex
labels = ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
critValues = []
for i in labels:
    minValue = dataset_x[i].min()
    maxValue = dataset_x[i].max()
    critValues.append([minValue, maxValue])

scaler = preprocessing.MinMaxScaler()
dataset_x = scaler.fit_transform(dataset_x)
scaled_df = pd.DataFrame(dataset_x, columns=['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare'])

dataset_y = np.array(dataset_y)

from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1=train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=3)


model = LinearRegression()
model.fit(X_train1, Y_train1)

y_pred1 = model.predict(X_test1)


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(Y_test1, y_pred1))

from sklearn.metrics import r2_score
print(r2_score(Y_test1, y_pred1))

from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(Y_test1, y_pred1))

import pickle
with open('titanic_linear', 'wb') as pkl:
    pickle.dump(model, pkl)