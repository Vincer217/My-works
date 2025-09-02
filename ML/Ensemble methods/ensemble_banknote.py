from matplotlib import pyplot as plt
from pathlib import Path 
from sys import path
from numpy import array 
from pandas import read_csv, DataFrame
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, recall_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split



script_dir = Path(path[0])
data = read_csv(script_dir/'banknote-auth.csv')

# Графический анализ. Отбор зависимых переменных 
# fig = plt.figure(figsize=(20,5))
# axs = fig.subplots(1,6)


# for i, el in enumerate(combinations(data.loc[:, :'entropy'], 2)):
    # axs[i].scatter(data[el[0]], data[el[1]])
    # axs[i].set(
        # xticks=[],
        # yticks=[],
        # xlabel=el[0],
        # ylabel=el[1]
        # )
   
    
# fig.show()    

# 0 — поддельная — отрицательный
# 1 — подлинная  — положительный

# Разделение на тестовую и обучающую выборки 
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
        data.loc[:, :'entropy'], 
        data['class'],
        test_size=1/3,
        random_state=1
        )

# ==================================RandomForest==================================
print('RandomForest', end='\n\n')

# Создание модели и подбор гиперпараметров 
model_rforest = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    n_jobs=-1,
    # class_weight='balanced'
)

model_rforest.fit(x_train_1, y_train_1)
y_pred_rforest = model_rforest.predict(x_test_1)
conf_matrix_rforest = confusion_matrix(y_pred_rforest, y_test_1)

# Визуализация
ConfusionMatrixDisplay(conf_matrix_rforest).plot(cmap='inferno').ax_.set_title('Confusion Matrix for RandomForest Model')

# Количество значений в каждом классе тестовой выборки целевой переменной       
# >>> y_test_1.value_counts()
# class
# 0    264
# 1    194 

print(f'accuracy = {accuracy_score(y_test_1, y_pred_rforest):.2f}'
      f'\nrecall = {recall_score(y_test_1, y_pred_rforest):.2f}'  
      f'\nprecision = {precision_score(y_test_1, y_pred_rforest):.2f}',
      end='\n\n'      
      ) 

# ==================================Bagging==================================
print('Bagging', end='\n\n')

# Создание ансамбля на основе модели логистической регрессии и подбор гиперпараметров          
model_bagging = BaggingClassifier(estimator=LogisticRegression(), n_estimators=50, n_jobs=-1) 
model_bagging.fit(x_train_1, y_train_1)
y_pred_bagging = model_bagging.predict(x_test_1)
        
conf_matrix_bagging = confusion_matrix(y_test_1, y_pred_bagging)

ConfusionMatrixDisplay(conf_matrix_bagging).plot(cmap='inferno').ax_.set_title('Confusion Matrix for Bagging Model')       
       
print(f'accuracy = {accuracy_score(y_test_1, y_pred_bagging):.2f}'
      f'\nrecall = {recall_score(y_test_1, y_pred_bagging):.2f}'  
      f'\nprecision = {precision_score(y_test_1, y_pred_bagging):.2f}', 
      end='\n\n'
      )   
 
# ==================================Stacking==================================
print('Stacking', end='\n\n')

# Создание первого слоя стэка и подбор гиперпараметров каждой модели 
model_1st_lvl = [
    LogisticRegression(),
    RidgeClassifier(),
    DecisionTreeClassifier(max_depth=5),
]

# Создание второго слоя стэка и подбор гиперпараметров
model_2nd_lvl = KNeighborsClassifier(n_neighbors=3)

# Предсказания первого слоя стэка 
pred_1st_lvl = []
# Точность моделей первого уровня
accuracy_1st_lvl = []

# Обучение моделей первого уровня 
for model in model_1st_lvl:
    model.fit(x_train_1, y_train_1)
    y_pred_1st_lvl = model.predict(x_test_1) 
    pred_1st_lvl.append(y_pred_1st_lvl)
    score_1st_lvl = round(accuracy_score(y_test_1, y_pred_1st_lvl), 2)
    accuracy_1st_lvl.append(score_1st_lvl)
    print(f'accuracy {str(model)} = {score_1st_lvl}') 
 
    
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

accuracy_stacking = round(accuracy_score(y_test_2, y_pred_2nd_lvl), 2)

conf_matrix_stacking = confusion_matrix(y_test_2, y_pred_2nd_lvl)

ConfusionMatrixDisplay(conf_matrix_stacking).plot(cmap='inferno').ax_.set_title('Confusion Matrix for Stacking Model')
 
print(f'\naccuracy {str(model_2nd_lvl)} = {accuracy_stacking}'
      f'\nrecall = {recall_score(y_test_2, y_pred_2nd_lvl):.2f}'  
      f'\nprecision = {precision_score(y_test_2, y_pred_2nd_lvl):.2f}', 
      end='\n\n'
      )

plt.show() 

# RandomForest

# accuracy = 0.99
# recall = 1.00
# precision = 0.97

# Bagging

# accuracy = 0.99
# recall = 0.99
# precision = 0.99

# Stacking

# accuracy LogisticRegression() = 0.99
# accuracy RidgeClassifier() = 0.98
# accuracy DecisionTreeClassifier(max_depth=5) = 0.99

# accuracy KNeighborsClassifier(n_neighbors=3) = 0.99
# recall = 0.97
# precision = 1.00      