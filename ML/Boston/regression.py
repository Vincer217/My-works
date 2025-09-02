from matplotlib import pyplot as plt
from matplotlib import rcParams
from numpy import ones
from pandas import read_csv, DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pathlib import Path
from random import seed, choice
from sys import path

rcParams['text.color'] = '#ddd'
rcParams['axes.labelcolor'] = '#ddd'


dir_path = Path(path[0])
data_path = dir_path / 'boston.csv'

data = read_csv(data_path, comment='#')

# >>> data.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 506 entries, 0 to 505
# Data columns (total 14 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   CRIM     506 non-null    float64
#  1   ZN       506 non-null    float64
#  2   INDUS    506 non-null    float64
#  3   CHAS     506 non-null    float64
#  4   NOX      506 non-null    float64
#  5   RM       506 non-null    float64
#  6   AGE      506 non-null    float64
#  7   DIS      506 non-null    float64
#  8   RAD      506 non-null    float64
#  9   TAX      506 non-null    float64
#  10  PTRATIO  506 non-null    float64
#  11  B        506 non-null    float64
#  12  LSTAT    506 non-null    float64
#  13  MEDV     506 non-null    float64
# dtypes: float64(14)
# memory usage: 55.5 KB

# >>> data['MEDV'].describe().round(2)
# count    506.00
# mean      22.53
# std        9.20
# min        5.00
# 25%       17.02
# 50%       21.20
# 75%       25.00
# max       50.00
# Name: MEDV, dtype: float64

# Графический анализ целевой переменной 
# fig = plt.figure(figsize=(5, 5))
# axs = fig.subplots()

# axs.scatter(data['MEDV'].index, data['MEDV'])

# fig.show()


corr_matrix_pearson = data.corr('pearson').round(2) # линейная корреляция 
corr_matrix_spearman = data.corr('spearman').round(2) # ранговая корреляция  


# fig = plt.figure(figsize=(28, 28))
# axs = fig.subplots(14, 14)
# 
# Корреляции переменных между собой
# for i, col1 in enumerate(data):
#     # for j, col2 in enumerate(data.columns[data.columns != col1]):
#     for j, col2 in enumerate(data):
#         axs[i][j].scatter(data[col1], data[col2], s=7)
#         axs[i][j].text(
#             data[col1].max(), 
#             data[col2].max(),
#             f'p={corr_matrix_pearson.loc[col1, col2]}\n'
#             f's={corr_matrix_spearman.loc[col1, col2]}',
#             horizontalalignment='right',
#             verticalalignment='top',
#         )
#         axs[i][j].set(
#             xticks=[], 
#             yticks=[],
#             xlabel=col1,
#             ylabel=col2,
#         )
# 
# fig.savefig(dir_path / 'boston_14x14_graphs.png', dpi=200)


# print(
#     'до отбраковки выбросов',
#     corr_matrix_pearson,
#     corr_matrix_spearman,
#     sep='\n\n',
#     end='\n\n',
# )

# Удаление выбросов 
data_out = data.loc[data['MEDV'] != data['MEDV'].max()]

print(
    'после отбраковки выбросов',
    # data_out.corr('pearson').round(2),
    data_out.corr('spearman').round(2),
    sep='\n\n',
    end='\n\n',
)


X = data_out.loc[:, ['CRIM', 'INDUS', 'RM', 'AGE', 'LSTAT']] # признаки, которые влияют на целевую переменную (корреляция больше 0.7) 
Y = data_out['MEDV']

# Разделение на тестовую и обучающую выборку в ручную

test_rate = 0.2
test_len = int(X.shape[0] * test_rate)
train_len = X.shape[0] - test_len

x_train, y_train, x_test, y_test = {}, {}, {}, {}

seed(2) # зерно для псевдо-случайного разбиения 

# Перемешиваем данные для обучающей выборки 
for _ in range(train_len):
    while True:
        rand_index = choice(X.index)
        if rand_index in x_train:
            continue
        break
    x_train[rand_index] = X.loc[rand_index]
    y_train[rand_index] = Y.loc[rand_index]

x_train = DataFrame(x_train).transpose()
y_train = Series(y_train).transpose()

# Перемешиваем данные для тестовой выборки
for rand_index in set(X.index) - set(x_train.index):
    x_test[rand_index] = X.loc[rand_index]
    y_test[rand_index] = Y.loc[rand_index]

x_test = DataFrame(x_test).transpose()
y_test = Series(y_test).transpose()

# x_train, x_test, y_train, y_test = train_test_split(
#     X, Y,
#     test_size=0.2,
#     random_state=17,
# )


# Создание модели
model = LinearRegression()

# Обучение модели
model.fit(x_train, y_train)

# Вычисление предсказанных значений для тестовой выборки
y_pred = model.predict(x_test)
# Оценка эффективности с помощью метрик R-квадрат и среднеквадратичная ошибка
R2 = r2_score(y_test, y_pred)
RMSE = (sum((y_test - y_pred)**2) / y_test.shape[0])**.5

print(f'R2 = {R2:.3f}\nRMSE = {RMSE:.1f}')

# R2 = 0.726
# RMSE = 4.4
