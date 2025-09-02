"""
Кластеризация данных и выполнение задачи регрессии на линейный кластер
"""

from numpy import load, linspace
from numpy.random import normal
from matplotlib import pyplot as plt 
from sys import path 
from pathlib import Path 
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split

data_dir = Path(path[0]) 
data = load(data_dir/'data1.npz')

# plt.figure(figsize=(10, 6))
# plt.scatter(data['x'], data['y'], color='blue', alpha=0.6, label="Исходные данные")
# plt.title("Визуализация данных: линейный и синусоидальный компоненты")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.grid(True)
# plt.legend()
# plt.show()

# Данные
X = data['x'].reshape(-1, 1)
y = data['y']

# thresholds = linspace(0.5, 1.5, 10)  # Диапазон порогов
# fig, axs = plt.subplots(2,5, figsize=(19.2,10.8))

# Выбор наилучшего residual_threshold
# for i in range(10):

    # ransac = RANSACRegressor(
                            # residual_threshold=thresholds[i],
                            # max_trials=1000,
                            # random_state=20
                            # )
    # ransac.fit(X, y)
    
    # inlier_mask = ransac.inlier_mask_ 
    # outlier_mask = ~inlier_mask       
    
    # line = axs[i//5][i%5].scatter(X[inlier_mask], y[inlier_mask], color='blue')
    # sinus = axs[i//5][i%5].scatter(X[outlier_mask], y[outlier_mask], color='red')
    # axs[i//5][i%5].set_title(f'residual_threshold = {round(thresholds[i],2)}')
    
# line.set_label('Линейный кластер')    
# sinus.set_label('Синусоидальный кластер')    
# fig.legend()
# fig.savefig(data_dir/'residual_threshold.png', dpi=100)

# Устойчивая регрессия
ransac = RANSACRegressor(
                        residual_threshold=0.61,
                        max_trials=1000,
                        random_state=20
                        ).fit(X, y)
                        
inlier_mask = ransac.inlier_mask_  # Маска "нормальных" точек (линейный кластер)
outlier_mask = ~inlier_mask       # Маска выбросов (синусоидальный кластер)

# Линейный кластер
x_line = X[inlier_mask] 
y_line = y[inlier_mask] 

# Получаем правильную линию тренда
line_X = linspace(X.min(), X.max(), 100).reshape(-1, 1)
line_y = ransac.predict(line_X)

# Визуализация
plt.figure(figsize=(19.2,10.8))
plt.scatter(x_line, y_line, color='blue', label='Линейный кластер')
plt.scatter(X[outlier_mask], y[outlier_mask], color='red', label='Синусоидальный кластер')
plt.plot(line_X, line_y, color='black', linewidth=3, label='Устойчивый тренд')
plt.legend()
plt.title('RANSAC: Корректное разделение кластеров\nresidual_threshold = 0.61')
plt.savefig(data_dir/'Clusters.png', dpi=100)

# =======================================================Regression====================================================

x_train, x_test, y_train, y_test = train_test_split(x_line, y_line, test_size=0.2, random_state=7)

model = LinearRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

# Визуализация
plt.figure(figsize=(19.2,10.8))
plt.scatter(x_train, y_train)
plt.plot(x_test, y_pred, color='black')
plt.title(f'Линия регрессии\ny = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
plt.savefig(data_dir/'Lin_Reg.png', dpi=100)

print(f'rmse = {rmse(y_test,y_pred):.2f}')
print(f'r2_score = {r2_score (y_test,y_pred):.2f}')

# rmse = 0.30
# r2_score = 0.99