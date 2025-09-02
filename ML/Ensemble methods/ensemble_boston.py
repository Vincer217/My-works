from pandas import read_csv
from pathlib import Path 
from sys import path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from numpy import array, corrcoef
from pandas import DataFrame


script_dir = Path(path[0])

data = read_csv(script_dir/'boston.csv', comment='#')

# Удаление выбросов после разведочного анализа данных 
data = data.drop(index=data[data['MEDV'] == 50].index)

# corr_matrix_pearson = data.corr('pearson').round(2)
# corr_matrix_spearman = data.corr('spearman').round(2)
# 
# Графический анализ 
# fig = plt.figure(figsize=(50, 8))
# axs = fig.subplots(1,13)
# 
# for i, el in enumerate(data.columns[:13]):
#     axs[i].scatter(data[el], data['MEDV'])
#     axs[i].set_title(f'p={corr_matrix_pearson.loc[el, 'MEDV']}\n'
#                      f's={corr_matrix_spearman.loc[el,'MEDV']}'
#     )
#     axs[i].set(
#         xticks=[],
#         yticks=[],
#         xlabel=el,
#         ylabel='MEDV'
#     )
   

# fig.savefig(script_dir/'boston_corr.png', dpi=100)


# Разделение на тестовую и обучающую выборки
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
            data.loc[:, ['CRIM', 'INDUS', 'RM', 'AGE', 'LSTAT']] , # отобранные зависимые переменные 
            data['MEDV'],
            test_size=0.2,
            random_state=1
)

# ==================================Stacking==================================
print('Stacking', end='\n\n')

# Создание первого слоя стэка и подбор гиперпараметров каждой модели
model_1st_lvl = [
    KNeighborsRegressor(n_neighbors=3),
    Ridge(),
    DecisionTreeRegressor(max_depth=5),
]

# Создание второго слоя стэка
model_2nd_lvl = LinearRegression()

# Предсказания первого слоя стэка
pred_1st_lvl = []
# Коэфициент детерминации моделей первого уровня
r2_1st_lvl = []

# Обучение моделей первого уровня
for model in model_1st_lvl:
    model.fit(x_train_1, y_train_1)
    y_pred_1st_lvl = model.predict(x_test_1) 
    pred_1st_lvl.append(y_pred_1st_lvl)
    score_1st_lvl = round(r2_score(y_test_1, y_pred_1st_lvl), 2)
    r2_1st_lvl.append(score_1st_lvl)
    print(f'r2 {str(model)} = {score_1st_lvl}') 
 
    
pred_1st_lvl = DataFrame(array(pred_1st_lvl).T) 

# Разделение на тестовую и обучающую выборки для второго слоя стэка    
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(
        pred_1st_lvl,
        y_test_1,
        test_size=0.2,
        random_state=1
        )

   
model_2nd_lvl.fit(x_train_2, y_train_2)
y_pred_2nd_lvl = model_2nd_lvl.predict(x_test_2)

# Коэфициент детерминации
r2_stacking = round(r2_score(y_test_2, y_pred_2nd_lvl), 2)

print(f'\nr2 {str(model_2nd_lvl)} = {r2_stacking}'
      f'\nRMSE = {round(root_mean_squared_error(y_test_2, y_pred_2nd_lvl), 2)}', 
      end='\n\n'
      )

# ==================================Bagging==================================
print('Bagging', end='\n\n')

# Создание ансамбля на основе модели дерева решений и подбор гиперпараметров
model_bagging = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=100, n_jobs=-1)
model_bagging.fit(x_train_1, y_train_1)
y_pred_bagging = model_bagging.predict(x_test_1)

# Коэфициент детерминации
r2_bagging = round(r2_score(y_test_1, y_pred_bagging), 2)
  
print(f'r2 {str(model_bagging)} = {r2_bagging}'
      f'\nRMSE = {round(root_mean_squared_error(y_test_1, y_pred_bagging), 2)}',
      end='\n\n'
      )


# ==================================RandomForest==================================
print('RandomForest', end='\n\n')

# Создание модели и подбор гиперпараметров 
model_rforest = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=1)
model_rforest.fit(x_train_1, y_train_1)
y_pred_rforest = model_rforest.predict(x_test_1)

# Коэфициент детерминации
r2_rforest = round(r2_score(y_test_1, y_pred_rforest), 2)

print(f'r2 {str(model_rforest)} = {r2_rforest}'
      f'\nRMSE = {round(root_mean_squared_error(y_test_1, y_pred_rforest), 2)}'
)

# ==================================Visualization==================================

fig, axs = plt.subplots(1, 3, figsize=(12,5))

# Угловой коэффициент линии регрессии 
slope_stacking = corrcoef(y_test_2, y_pred_2nd_lvl)[0, 1] *  y_pred_2nd_lvl.std() / y_test_2.std()
slope_bagging = corrcoef(y_test_2, y_pred_2nd_lvl)[0, 1] *  y_pred_2nd_lvl.std() / y_test_2.std()
slope_rforest = corrcoef(y_test_2, y_pred_2nd_lvl)[0, 1] *  y_pred_2nd_lvl.std() / y_test_2.std()

axs[0].scatter(y_test_2, y_pred_2nd_lvl)
axs[0].axline((y_test_2.mean(), y_pred_2nd_lvl.mean()), slope=slope_stacking, c='#f54242')
axs[0].set(xlabel='Фактические значения', ylabel='Предсказанные значения', title=f'Stacking\nr2 = {r2_stacking}')

axs[1].scatter(y_test_1, y_pred_bagging)
axs[1].axline((y_test_1.mean(), y_pred_bagging.mean()), slope=slope_bagging, c='#f54242')
axs[1].set(xlabel='Фактические значения', ylabel='Предсказанные значения', title=f'Bagging\nr2 = {r2_bagging}')

axs[2].scatter(y_test_1, y_pred_rforest)
axs[2].axline((y_test_1.mean(), y_pred_rforest.mean()), slope=slope_rforest, c='#f54242')
axs[2].set(xlabel='Фактические значения', ylabel='Предсказанные значения', title=f'RandomForest\nr2 = {r2_rforest}')


fig.show()

# Stacking
# 
# r2 KNeighborsRegressor(n_neighbors=3) = 0.58
# r2 Ridge() = 0.67
# r2 DecisionTreeRegressor(max_depth=5) = 0.79
# 
# r2 LinearRegression() = 0.93
# RMSE = 2.42
# 
# Bagging
# 
# r2 BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6), n_estimators=100, n_jobs=-1) = 0.8
# RMSE = 3.33
# 
# RandomForest
# 
# r2 RandomForestRegressor(max_depth=6, n_estimators=60, random_state=1) = 0.8
# RMSE = 3.34