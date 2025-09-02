from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split

from pathlib import Path
from sys import path


dir_path = Path(path[0])


data_all = read_csv(dir_path / 'banknote-auth.csv')

# >>> data_all.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1372 entries, 0 to 1371
# Data columns (total 5 columns):
#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   variance  1372 non-null   float64
#  1   skewness  1372 non-null   float64
#  2   curtosis  1372 non-null   float64
#  3   entropy   1372 non-null   float64
#  4   class     1372 non-null   int64
# dtypes: float64(4), int64(1)
# memory usage: 53.7 KB


# >>> data_all['class'].value_counts()
# class
# 0    762 - поддельные купюры 
# 1    610 - настоящие купюры
# Name: count, dtype: int64

# Разделение на тестовую и обучающую выборки
x_train, x_test, y_train, y_test = train_test_split(
    data_all.loc[:, ['variance', 'skewness', 'curtosis', 'entropy']],
    data_all['class'],
    test_size=1/3,
    random_state=1,
)

# >>> y_test.value_counts()
# class
# 0    264
# 1    194
# Name: count, dtype: int64

# 0 — поддельная — отрицательный
# 1 — подлинная  — положительный

# Инициализация модели 
model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

conf_matr = confusion_matrix(y_test, y_pred)

# >>> conf_matr
# array([[262,   2],
#        [  2, 192]])

(TN, FP), (FN, TP) = conf_matr

# Общая точность (корректность) модели
accuracy = (TN + TP) / (TN + FP + FN + TP)
# Доля корректно сделанных прогнозов для отрицательного класса
specificity = TN / (TN + FP)
# Доля корректно сделанных прогнозов среди предсказаний положительного класса
precision = TP / (FP + TP)
# Доля истинно положительных среди всех положительных
recall = TP / (FN + TP)

# Среднее гармоническое значение точности и полноты
f1 = 2 * precision * recall / (precision + recall)
# Приоритет в сторону метрики precision
fbeta = fbeta_score(y_test, y_pred, beta=0.5)

print(
    f'{accuracy=:.1%}',
    f'{specificity=:.1%}',
    f'{precision=:.1%}',
    f'{recall=:.1%}',
    f'{f1=:.1%}',
    f'{fbeta=:.1%}',
    sep='\n'
)
# accuracy=99.1%
# specificity=99.2%
# precision=99.0%
# recall=99.0%
# f1=99.0%
# fbeta=99.0%

