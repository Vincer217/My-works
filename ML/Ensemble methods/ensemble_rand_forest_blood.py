from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.io.arff import loadarff
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from pathlib import Path
from sys import path


script_dir = Path(path[0])
with open(script_dir / 'blood.arff', encoding='utf-8') as filein:
    data_raw = loadarff(filein)

blood = DataFrame(data_raw[0])

# Название столбцов
blood.columns = ['recency', 'frequency', 'monetary', 'time', 'donated']

# Обозначения категориальной переменной
blood.loc[blood['donated'] == b'1', 'donated'] = 0
blood.loc[blood['donated'] == b'2', 'donated'] = 1

blood = blood.astype(dtype=int)

# Разделение на обучающую и тестовую выборки. Исключена переменная 'monetary', так как она имеет линейную зависимость с 'frequency'
x_train, x_test, y_train, y_test = train_test_split(
    blood.loc[:, ['recency', 'frequency', 'time']],
    blood['donated'],
    test_size=0.2,
    random_state=1
)

# Инициализация модели и подбор гиперпараметров 
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    n_jobs=-1
)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

conf = confusion_matrix(y_test, y_predict)
print(conf)
# ConfusionMatrixDisplay(conf).plot(cmap='inferno').figure_.show()
print(
    f'\naccuracy = {accuracy_score(y_test, y_predict):.2f}'
    f'\nrecall = {recall_score(y_test, y_predict):.2f}\n'
)


