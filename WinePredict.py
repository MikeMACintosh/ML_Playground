'''
Имеется набор данных о трех сортах красных вин,
датасет содержит 178 строк и 13 столбцов-параметров вина.
Попробуем на основе имеющихся статистических данных определять сорт вина.

Построим нейронную сеть с 3-мя скрытыми слоями и 13-ю входами(количество параметров).
'''

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

os.chdir('C:/Users/User/Downloads')
columns = ['Sort', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
           'Magnesium', 'Total phenols', 'Flavanoids',
           'Nonflavanoid phenols', 'Proanthocyanins','Color intensity',
           'Hue', 'OD280/OD315 of diluted wines', 'Proline'
           ]
wine = pd.read_csv('wine.data', names=columns)

wine['Sort'].value_counts(normalize=True)
y = wine['Sort']
X = wine.drop('Sort', axis=1)

# Разбиваем на обучающую и тестовую выборку.
# Задаём "зерно" датчика случайных чисел, определим объем тестовой выборки:
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345, test_size=0.33)

X_train.head()
X_test.head()
y_train.head()
y_test.head()

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
y_train_bin = np_utils.to_categorical(y_train)
y_test_bin = np_utils.to_categorical(y_test)
model = Sequential()
model.add(Dense(9, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train_bin, epochs=300, batch_size=10)
