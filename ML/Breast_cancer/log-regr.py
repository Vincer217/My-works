from matplotlib import pyplot as plt, rcParams
from pandas import DataFrame, Series
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split

from pathlib import Path
from sys import path


dir_path = Path(path[0])


data_raw = load_breast_cancer()

data = DataFrame(data_raw['data'], columns=data_raw['feature_names'])
target = Series(data_raw['target'], name='target')

# Все данные вместе с целевой переменной
data_all = DataFrame(
      dict(zip(
          data_raw['feature_names'], 
          data_raw['data'].T
      )) 
    | {'target': data_raw['target']}
) 

# >>> data_all.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 569 entries, 0 to 568
# Data columns (total 31 columns):
#  #   Column                   Non-Null Count  Dtype
# ---  ------                   --------------  -----
#  0   mean radius              569 non-null    float64
#  1   mean texture             569 non-null    float64
#  2   mean perimeter           569 non-null    float64
#  3   mean area                569 non-null    float64
#  4   mean smoothness          569 non-null    float64
#  5   mean compactness         569 non-null    float64
#  6   mean concavity           569 non-null    float64
#  7   mean concave points      569 non-null    float64
#  8   mean symmetry            569 non-null    float64
#  9   mean fractal dimension   569 non-null    float64
#  10  radius error             569 non-null    float64
#  11  texture error            569 non-null    float64
#  12  perimeter error          569 non-null    float64
#  13  area error               569 non-null    float64
#  14  smoothness error         569 non-null    float64
#  15  compactness error        569 non-null    float64
#  16  concavity error          569 non-null    float64
#  17  concave points error     569 non-null    float64
#  18  symmetry error           569 non-null    float64
#  19  fractal dimension error  569 non-null    float64
#  20  worst radius             569 non-null    float64
#  21  worst texture            569 non-null    float64
#  22  worst perimeter          569 non-null    float64
#  23  worst area               569 non-null    float64
#  24  worst smoothness         569 non-null    float64
#  25  worst compactness        569 non-null    float64
#  26  worst concavity          569 non-null    float64
#  27  worst concave points     569 non-null    float64
#  28  worst symmetry           569 non-null    float64
#  29  worst fractal dimension  569 non-null    float64
#  30  target                   569 non-null    int64
# dtypes: float64(30), int64(1)
# memory usage: 137.9 KB


# >>> target.value_counts()
# target
# 1    357
# 0    212
# Name: count, dtype: int64

# 0 — злокачественные
# 1 — доброкачественные


# >>> data.describe().transpose().round(2)
#                          count    mean     std     min     25%     50%      75%      max
# mean radius              569.0   14.13    3.52    6.98   11.70   13.37    15.78    28.11
# mean texture             569.0   19.29    4.30    9.71   16.17   18.84    21.80    39.28
# mean perimeter           569.0   91.97   24.30   43.79   75.17   86.24   104.10   188.50
# mean area                569.0  654.89  351.91  143.50  420.30  551.10   782.70  2501.00
# mean smoothness          569.0    0.10    0.01    0.05    0.09    0.10     0.11     0.16
# mean compactness         569.0    0.10    0.05    0.02    0.06    0.09     0.13     0.35
# mean concavity           569.0    0.09    0.08    0.00    0.03    0.06     0.13     0.43
# mean concave points      569.0    0.05    0.04    0.00    0.02    0.03     0.07     0.20
# mean symmetry            569.0    0.18    0.03    0.11    0.16    0.18     0.20     0.30
# mean fractal dimension   569.0    0.06    0.01    0.05    0.06    0.06     0.07     0.10
# radius error             569.0    0.41    0.28    0.11    0.23    0.32     0.48     2.87
# texture error            569.0    1.22    0.55    0.36    0.83    1.11     1.47     4.88
# perimeter error          569.0    2.87    2.02    0.76    1.61    2.29     3.36    21.98
# area error               569.0   40.34   45.49    6.80   17.85   24.53    45.19   542.20
# smoothness error         569.0    0.01    0.00    0.00    0.01    0.01     0.01     0.03
# compactness error        569.0    0.03    0.02    0.00    0.01    0.02     0.03     0.14
# concavity error          569.0    0.03    0.03    0.00    0.02    0.03     0.04     0.40
# concave points error     569.0    0.01    0.01    0.00    0.01    0.01     0.01     0.05
# symmetry error           569.0    0.02    0.01    0.01    0.02    0.02     0.02     0.08
# fractal dimension error  569.0    0.00    0.00    0.00    0.00    0.00     0.00     0.03
# worst radius             569.0   16.27    4.83    7.93   13.01   14.97    18.79    36.04
# worst texture            569.0   25.68    6.15   12.02   21.08   25.41    29.72    49.54
# worst perimeter          569.0  107.26   33.60   50.41   84.11   97.66   125.40   251.20
# worst area               569.0  880.58  569.36  185.20  515.30  686.50  1084.00  4254.00
# worst smoothness         569.0    0.13    0.02    0.07    0.12    0.13     0.15     0.22
# worst compactness        569.0    0.25    0.16    0.03    0.15    0.21     0.34     1.06
# worst concavity          569.0    0.27    0.21    0.00    0.11    0.23     0.38     1.25
# worst concave points     569.0    0.11    0.07    0.00    0.06    0.10     0.16     0.29
# worst symmetry           569.0    0.29    0.06    0.16    0.25    0.28     0.32     0.66
# worst fractal dimension  569.0    0.08    0.02    0.06    0.07    0.08     0.09     0.21

# Нормализация 
data_norm = (data - data.describe().loc['mean']) / data.describe().loc['std']


# >>> data_norm.describe().transpose().round(2)
#                          count  mean  std   min   25%   50%   75%    max
# mean radius              569.0  -0.0  1.0 -2.03 -0.69 -0.21  0.47   3.97
# mean texture             569.0   0.0  1.0 -2.23 -0.73 -0.10  0.58   4.65
# mean perimeter           569.0  -0.0  1.0 -1.98 -0.69 -0.24  0.50   3.97
# mean area                569.0  -0.0  1.0 -1.45 -0.67 -0.29  0.36   5.25
# mean smoothness          569.0  -0.0  1.0 -3.11 -0.71 -0.03  0.64   4.77
# mean compactness         569.0   0.0  1.0 -1.61 -0.75 -0.22  0.49   4.56
# mean concavity           569.0   0.0  1.0 -1.11 -0.74 -0.34  0.53   4.24
# mean concave points      569.0  -0.0  1.0 -1.26 -0.74 -0.40  0.65   3.92
# mean symmetry            569.0   0.0  1.0 -2.74 -0.70 -0.07  0.53   4.48
# mean fractal dimension   569.0   0.0  1.0 -1.82 -0.72 -0.18  0.47   4.91
# radius error             569.0   0.0  1.0 -1.06 -0.62 -0.29  0.27   8.90
# texture error            569.0  -0.0  1.0 -1.55 -0.69 -0.20  0.47   6.65
# perimeter error          569.0  -0.0  1.0 -1.04 -0.62 -0.29  0.24   9.45
# area error               569.0  -0.0  1.0 -0.74 -0.49 -0.35  0.11  11.03
# smoothness error         569.0  -0.0  1.0 -1.77 -0.62 -0.22  0.37   8.02
# compactness error        569.0   0.0  1.0 -1.30 -0.69 -0.28  0.39   6.14
# concavity error          569.0   0.0  1.0 -1.06 -0.56 -0.20  0.34  12.06
# concave points error     569.0   0.0  1.0 -1.91 -0.67 -0.14  0.47   6.64
# symmetry error           569.0   0.0  1.0 -1.53 -0.65 -0.22  0.36   7.07
# fractal dimension error  569.0  -0.0  1.0 -1.10 -0.58 -0.23  0.29   9.84
# worst radius             569.0  -0.0  1.0 -1.73 -0.67 -0.27  0.52   4.09
# worst texture            569.0   0.0  1.0 -2.22 -0.75 -0.04  0.66   3.88
# worst perimeter          569.0  -0.0  1.0 -1.69 -0.69 -0.29  0.54   4.28
# worst area               569.0   0.0  1.0 -1.22 -0.64 -0.34  0.36   5.92
# worst smoothness         569.0  -0.0  1.0 -2.68 -0.69 -0.05  0.60   3.95
# worst compactness        569.0  -0.0  1.0 -1.44 -0.68 -0.27  0.54   5.11
# worst concavity          569.0   0.0  1.0 -1.30 -0.76 -0.22  0.53   4.70
# worst concave points     569.0   0.0  1.0 -1.74 -0.76 -0.22  0.71   2.68
# worst symmetry           569.0   0.0  1.0 -2.16 -0.64 -0.13  0.45   6.04
# worst fractal dimension  569.0  -0.0  1.0 -1.60 -0.69 -0.22  0.45   6.84

# Усреднение значений по классам 
mean_0 = data_norm.loc[target == 0].mean()
mean_1 = data_norm.loc[target == 1].mean()

groupped = DataFrame({
    'mean 0': mean_0,
    'mean 1': mean_1,
    'diff': abs(mean_0 - mean_1),
}).sort_values(by='diff', ascending=False)
# 'diff' - показывает наболее зависимые количественные переменные от соответствующего категориального значения 

# >>> groupped.round(2)
#                          mean 0  mean 1  diff
# worst concave points       1.03   -0.61  1.64
# worst perimeter            1.02   -0.60  1.62
# mean concave points        1.01   -0.60  1.60
# worst radius               1.01   -0.60  1.60
# mean perimeter             0.96   -0.57  1.53
# worst area                 0.95   -0.56  1.52
# mean radius                0.95   -0.56  1.51
# mean area                  0.92   -0.55  1.47
# mean concavity             0.90   -0.54  1.44
# worst concavity            0.86   -0.51  1.36
# mean compactness           0.77   -0.46  1.23
# worst compactness          0.77   -0.46  1.22
# radius error               0.74   -0.44  1.17
# perimeter error            0.72   -0.43  1.15
# area error                 0.71   -0.42  1.13
# worst texture              0.59   -0.35  0.94
# worst smoothness           0.55   -0.32  0.87
# worst symmetry             0.54   -0.32  0.86
# mean texture               0.54   -0.32  0.86
# concave points error       0.53   -0.31  0.84
# mean smoothness            0.46   -0.28  0.74
# mean symmetry              0.43   -0.25  0.68
# worst fractal dimension    0.42   -0.25  0.67
# compactness error          0.38   -0.23  0.61
# concavity error            0.33   -0.20  0.52
# fractal dimension error    0.10   -0.06  0.16
# smoothness error          -0.09    0.05  0.14
# mean fractal dimension    -0.02    0.01  0.03
# texture error             -0.01    0.01  0.02
# symmetry error            -0.01    0.01  0.01

# Разведочный анализ, исходя из которого, будет взято 10 признаков
# bins = 30 # количество интервалов
# 
# rcParams['axes.labelcolor'] = '#f8f8f2'
# 
# fig = plt.figure(figsize=(8, 4))
# axs = fig.subplots()
# 
# for var_name in groupped.index[:10]:
#     axs.clear()
#     axs.hist(
#         data_norm.loc[target == 0, var_name],
#         bins=bins,
#         alpha=0.5,
#         label='(0) злокачественная'
#     )
#     axs.hist(
#         data_norm.loc[target == 1, var_name],
#         bins=bins,
#         alpha=0.5,
#         label='(1) доброкачественная'
#     )
#     axs.set(
#         xlabel=var_name, 
#         ylabel='Количество значений в интервале',
#     )
#     axs.legend()
#     fig.savefig(dir_path / f'breast_cancer/{var_name}.png', dpi=150)

# Обработанные и отобранные данные 
X = data_norm.loc[:, groupped.index[:10]]

# Разделение на тестовую и обучающую выборки
x_train, x_test, y_train, y_test = train_test_split(
    X, target,
    test_size=0.2,
    random_state=1
)
# >>> y_test.value_counts()
# target
# 1    72
# 0    42
# Name: count, dtype: int64

# 0 — злокачественные   — отрицательный
# 1 — доброкачественные — положительный


model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

conf_matr = confusion_matrix(y_test, y_pred)

# >>> conf_matr
# array([[37,  5],
#        [ 0, 72]])
#  5 - это ошибки второго рода, то есть пять значений было предсказано как доброкачественные, хотя являются злокачественными (ложноположительные)

(TN, FP), (FN, TP) = conf_matr

accuracy = (TN + TP) / (TN + FN + FP + TP)
# accuracy = sum(conf_matr.diagonal()) / x_test.shape[0]

# Общая точность (корректность) модели
# >>> print(f'{accuracy:.1%}')
# 95.6%

specificity = TN / (TN + FP) # доля корректно сделанных прогнозов для отрицательного класса
# specificity = conf_matr[0,0] / sum(conf_matr[0])

# >>> print(f'{specificity:.1%}')
# 88.1%

# вероятность совершить ошибку второго рода (считая именно отрицательный класс более важным в контексте задачи)
# >>> print(f'{1 - specificity:.1%}')
# 11.9%

precision = TP / (FP + TP) # доля корректно сделанных прогнозов среди предсказаний положительного класса 
# precision = conf_matr[1,1] / sum(conf_matr[:, 1])

# >>> print(f'{precision:.1%}')
# 93.5%

recall = TP / (FN + TP) # доля истинно положительных среди всех положительных 
# recall = conf_matr[1,1] / sum(conf_matr[1])

# >>> print(f'{recall:.1%}')
# 100.0%

# вероятность совершить ошибку второго рода (считая положительный класс более важным в контексте задачи)
# >>> print(f'{1 - recall:.1%}')
# 0%

# Среднее гармоническое значение точности и полноты
f1 = 2 * precision * recall / (precision + recall)

# >>> print(f'{f1:.1%}')
# 96.6%

# Понижение значения параметра beta приводит к повышению важности метрики precision, а значит ложноположительных (FP) результатов, а значит отрицательного класса
fbeta = fbeta_score(y_test, y_pred, beta=0.001)

# >>> print(f'{fbeta:.1%}')
# 94.7%

print(
    f'{accuracy=:.1%}',
    f'{specificity=:.1%}',
    f'ошибок второго рода: {FP}',
    f'{precision=:.1%}',
    f'{recall=:.1%}',
    f'{f1=:.1%}',
    f'{fbeta=:.1%}',
    sep='\n'
)
# accuracy=95.6%
# specificity=88.1%
# ошибок второго рода: 5
# precision=93.5%
# recall=100.0%
# f1=96.6%
# fbeta=93.5%

