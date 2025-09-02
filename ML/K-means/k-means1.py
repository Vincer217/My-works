from matplotlib import pyplot as plt, rcParams
from numpy import array
from pandas import read_csv, DataFrame

from pathlib import Path
from random import uniform
from sys import path
from time import sleep


def euclid_line(point1, point2):
    """
    Вычисляет Евклидово расстояние между точками.
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2)**2 + (y1 - y2)**2)**.5


dir_path = Path(path[0])


rcParams['axes.labelcolor'] = '#eee'
rcParams['axes.labelsize'] = 16
rcParams['axes.titlecolor'] = '#eee'
rcParams['axes.titlesize'] = 18

colors = ['#ff8c00', '#6b8e23', '#deb887', '#800080', '#40e0d0']

plt.ion()

fig = plt.figure(figsize=(9, 9))
axs = fig.subplots()

# Данные с чётко заметными группами 
# data = read_csv(dir_path / 'test_clusters.csv', names=['x1', 'x2'])
# Смешанные данные 
data = read_csv(dir_path / 'test_clusters_mixed.csv', names=['x1', 'x2'])

# ====================== визуализация ======================
# Графический разведочный анализ данных, из которого выяснили, что групп будет 3 (для test_clusters)
axs.clear()
axs.set(
    xlabel='x1', 
    ylabel='x2',
    title='исходные данные \n'
)
axs.scatter(*data.values.T)
plt.draw()
plt.gcf().canvas.flush_events()
sleep(0.3)
# ====================== визуализация ======================


# выбор начального положения центроид по границам диапазона
# centroids = array([
#     (data['x1'].min(), data['x2'].min()),
#     (data['x1'].mean(), data['x2'].mean()),
#     (data['x1'].max(), data['x2'].max()),
# ])

# выбор случайного начального положения центроид
centroids = array([
    (uniform(data['x1'].min(), data['x1'].max()), uniform(data['x2'].min(), data['x2'].max())),
    (uniform(data['x1'].min(), data['x1'].max()), uniform(data['x2'].min(), data['x2'].max())),
    (uniform(data['x1'].min(), data['x1'].max()), uniform(data['x2'].min(), data['x2'].max())),
])
n = len(centroids)

# ====================== визуализация ======================
axs.set(
    xlabel='x1', 
    ylabel='x2',
    title='начальное положение центроид \n'
)
axs.scatter(*centroids.T, s=100, c='#c00', marker='^')
plt.draw()
plt.gcf().canvas.flush_events()
sleep(0.3)
# ====================== визуализация ======================

# Вручную прописанный алгоритм к-средних
for _ in range(7):

    cluster = []
    for point_i in data.values:
        lines = []
        for centr_k in centroids:
            # вычисление расстояния от каждой точки до каждой центроиды
            lines.append(euclid_line(point_i, centr_k))
        # выбор минимального из вычисленных расстояний
        cluster.append(array(lines).argmin()) # возвращает индекс 
    
    # ====================== визуализация ======================
    axs.clear()
    axs.set(
        xlabel='x1', 
        ylabel='x2',
        title='вычисление расстояний до центроид \n и выбор минимального расстояния'
    )
    
    # Назначение цвета для каждого кластера
    for i, point_cluster in enumerate(cluster):
        axs.scatter(*data.loc[i], c=colors[point_cluster])
    axs.scatter(*centroids.T, s=100, c=colors[:n], marker='^')
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(0.4)
    # ====================== визуализация ======================
    # Нахождение центров каждого кластера
    centroids = []
    for k in range(n):
        # Накладываем маску, чтобы вычленить значения, которые относятся к тому или иному кластеру 
        x1_k = data.loc[array(cluster) == k]['x1']
        x2_k = data.loc[array(cluster) == k]['x2']
        centroids.append((x1_k.mean(), x2_k.mean()))
    centroids = array(centroids) # новое положение центроид
    
    # ====================== визуализация ======================
    axs.set(
        xlabel='x1', 
        ylabel='x2',
        title='вычисление центров полученных кластеров \n'
    )
    axs.scatter(*centroids.T, s=200, c=colors[:n], marker='2')
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(0.4)

    axs.clear()
    axs.set(
        xlabel='x1', 
        ylabel='x2',
        title='новое положение центроид \n'
    )
    for i, point_cluster in enumerate(cluster):
        axs.scatter(*data.loc[i], c=colors[point_cluster])
    axs.scatter(*centroids.T, s=100, c='#c00', marker='^')
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(0.2)
    
    axs.clear()
    axs.set(
        xlabel='x1', 
        ylabel='x2',
        title='новое положение центроид \n'
    )
    axs.scatter(*data.values.T)
    axs.scatter(*centroids.T, s=100, c='#c00', marker='^')
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(0.2)
    # ====================== визуализация ======================


# ====================== визуализация ======================
axs.clear()
axs.set(
    xlabel='x1', 
    ylabel='x2',
    title='итоговое расположение центроид и кластеров \n'
)
for i, point_cluster in enumerate(cluster):
    axs.scatter(*data.loc[i], c=colors[point_cluster])
axs.scatter(*centroids.T, s=100, c=colors[:n], marker='^')
plt.draw()
plt.gcf().canvas.flush_events()
# ====================== визуализация ======================


data = DataFrame(data.to_dict() | {'cluster': {i: c for i, c in enumerate(cluster)}})


plt.ioff()
plt.show()

