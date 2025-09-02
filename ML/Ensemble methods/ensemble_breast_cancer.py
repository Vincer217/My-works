from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from pathlib import Path 
from sys import path
from numpy import array

dir_path = Path(path[0])

data = load_breast_cancer() # набор данных по раку молочной железы 

features = DataFrame(data['data'], columns=data['feature_names']) # признаки (зависимые переменные)

target = Series(data['target'], name='target') # целевая переменная

# >>> target.value_counts()
# target
# 1    357 - доброкачественная опухоль 
# 0    212 - злокачественная опухоль

features_norm = (features - features.describe().loc['mean']) / features.describe().loc['std'] # нормализация данных 

# Разделение признаков на классы и нахождение среднего значения для каждой переменной
mean_0 = features_norm.loc[target==0].mean()
mean_1 = features_norm.loc[target==1].mean()

# Разница между классами одинаковых переменных
diff_mean = DataFrame({
        'mean 0': mean_0,
        'mean 1': mean_1,
        'diff': abs(mean_0 - mean_1)
}).sort_values(by='diff', ascending=False)

bins = 30 # интервалы 

# Графический анализ и выбор признаков 
# fig, axs = plt.subplots(figsize=(12,8))

# for el in diff_mean.index[:12]:
    # axs.clear()
    # axs.hist(
        # features_norm.loc[target==0, el], 
        # bins=bins, 
        # alpha=0.5, 
        # label='(0) злокачественная'
        # )    
    # axs.hist(
        # features_norm.loc[target==1, el], 
        # bins=bins, 
        # alpha=0.5, 
        # label='(1) доброкачественная'
        # )
    # axs.set(xlabel=el, ylabel='Количество значений в интервале')
    # axs.set_title(f'diff = {round(diff_mean.loc[el ,'diff'], 2)}')    
    # axs.legend()
    # fig.savefig(dir_path/f'breast_cancer-{el}.png', dpi=150)    

# Разделение на тестовую и обучающую выборки    
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
        features_norm[diff_mean.index[:25]], 
        target, 
        test_size=0.2, 
        random_state=1
        ) 

# ==================================Bagging==================================
print('Bagging', end='\n\n')

# Создание ансамбля на основе модели логистической регрессии и подбор гиперпараметров          
model_bagging = BaggingClassifier(estimator=LogisticRegression(), n_estimators=50, n_jobs=-1) 
model_bagging.fit(x_train_1, y_train_1)
y_pred_bagging = model_bagging.predict(x_test_1)
        
conf_matrix_bagging = confusion_matrix(y_test_1, y_pred_bagging)

ConfusionMatrixDisplay(conf_matrix_bagging).plot(cmap='inferno').ax_.set_title('Confusion Matrix for Bagging Model')       
       
print(f'accuracy = {accuracy_score(y_test_1, y_pred_bagging):.2f}', end='\n\n')

# ==================================RandomForest==================================
print('RandomForest', end='\n\n')

# Создание модели и подбор гиперпараметров
model_rforest = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=1)
model_rforest.fit(x_train_1, y_train_1)
y_pred_rforest = model_rforest.predict(x_test_1)

conf_matrix_rforest = confusion_matrix(y_test_1, y_pred_rforest)

ConfusionMatrixDisplay(conf_matrix_rforest).plot(cmap='inferno').ax_.set_title('Confusion Matrix for RandomForest Model')       

print(f'accuracy = {accuracy_score(y_test_1, y_pred_rforest):.2f}', end='\n\n')

# ==================================Stacking==================================
print('Stacking', end='\n\n')

# Создание первого слоя стэка и подбор гиперпараметров каждой модели
model_1st_lvl = [
    LogisticRegression(),
    RidgeClassifier(),
    DecisionTreeClassifier(max_depth=5),
]

# Создание второго слоя стэка
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
 
print(f'\naccuracy {str(model_2nd_lvl)} = {accuracy_stacking}')

plt.show()
      
# Bagging

# accuracy = 0.98

# RandomForest

# accuracy = 0.96

# Stacking

# accuracy LogisticRegression() = 0.98
# accuracy RidgeClassifier() = 0.96
# accuracy DecisionTreeClassifier(max_depth=5) = 0.95

# accuracy KNeighborsClassifier(n_neighbors=3) = 1.0      