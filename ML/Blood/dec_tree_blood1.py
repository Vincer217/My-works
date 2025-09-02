from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.io.arff import loadarff
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix, 
    ConfusionMatrixDisplay,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from pathlib import Path
from sys import path


script_dir = Path(path[0])
with open(script_dir / 'blood.arff', encoding='utf-8') as filein:
    data_raw = loadarff(filein) # возвращает кортеж с данными и метаданными (атрибуты, связи и т.д.)

blood = DataFrame(data_raw[0])

blood.columns = ['recency', 'frequency', 'monetary', 'time', 'donated'] # меняем название переменных


# меняем обозначения в классе (целевой переменной)
blood.loc[blood['donated'] == b'1', 'donated'] = 0
blood.loc[blood['donated'] == b'2', 'donated'] = 1

blood = blood.astype(dtype=int)

blood.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 748 entries, 0 to 747
# Data columns (total 5 columns):
#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   recency    748 non-null    int64
#  1   frequency  748 non-null    int64
#  2   monetary   748 non-null    int64
#  3   time       748 non-null    int64
#  4   donated    748 non-null    int64
# dtypes: int64(5)
# memory usage: 29.3 KB

# >>> blood['donated'].value_counts()
# donated
# 0    570
# 1    177
# Name: count, dtype: int64

# 1 - пожертвовали кровь в марте 2007 (положительный)
# 0 - не пожертвовали (отрицательный)

corr_table = blood.corr().round(2)  # корреляция между переменными 

# Построение графиков для разведочного анализа данных 
fig = plt.figure(figsize=(15, 15))
axs = fig.subplots(*corr_table.shape)

for i, var_i in enumerate(corr_table.index):
    for j, var_j in enumerate(corr_table.columns):
        axs[i][j].scatter(blood[var_i], blood[var_j])
        axs[i][j].set_title(
            f'{var_i.upper()}—{var_j.upper()}\n'
            f'corr = {corr_table.loc[var_i, var_j]}'
        )
        axs[i][j].xaxis.set_visible(False)
        axs[i][j].yaxis.set_visible(False)

fig.savefig(script_dir / 'blood_corr.png', dpi=150)
# по результатам графического анализа — monetary не включать (за бесполезностью)

# Разделение на тестовую и обучающую выборки 
x_train, x_test, y_train, y_test = train_test_split(
    blood.loc[:, ['recency', 'frequency', 'time']],
    blood['donated'],
    test_size=0.2,
    random_state=1
)

# Инициализация модели и подбор гиперпараметров 
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=1
)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
conf = confusion_matrix(y_test, y_predict)
# >>> conf
# array([[94, 15],
#        [20, 21]])
ConfusionMatrixDisplay(conf).plot(cmap='inferno').figure_.show()
print(
    f'\naccuracy = {accuracy_score(y_test, y_predict):.2f}'
    f'\nrecall = {recall_score(y_test, y_predict):.2f}\n'
)
# accuracy = 0.77
# recall = 0.51

fig = plt.figure(figsize=(15, 12))
axs = fig.subplots()

plot_tree(model, ax=axs)

fig.savefig(script_dir / 'blood_tree.png', dpi=600)

