from matplotlib import pyplot as plt, rcParams, colormaps
from numpy import array
from numpy.random import default_rng
from pandas import read_csv, DataFrame

from itertools import combinations, pairwise
from pathlib import Path
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

colors = colormaps['tab10'].colors

anim_speed = 0.005 # скорость анимации 

plt.ion()

fig = plt.figure(figsize=(16, 8))
axs = fig.subplots(1, 2)

# Смешанные данные
data = read_csv(dir_path / 'test_clusters_mixed.csv', names=['x1', 'x2'])
wcss = [] # суммы внутрикластерных расстояний 

# ====================== визуализация ======================
# Графический разведочный анализ данных, из которого не получилось выяснить количество кластеров ввиду того, что данные плохо кластеризуемы
axs[1].clear()
axs[1].set(
    xlabel='x1', 
    ylabel='x2',
    title='исходные данные \n'
)
axs[1].scatter(*data.values.T)
plt.draw()
plt.gcf().canvas.flush_events()
sleep(anim_speed)
# ====================== визуализация ======================


# Проверка различного количества кластеров с помощью метода "локтя" 
for n in range(2, 6):
    
    # Выбор случайных точек для начального положения центроид
    centroids = default_rng().choice(data, n, replace=False)
    # ====================== визуализация ======================
    axs[1].set(
        xlabel='x1', 
        ylabel='x2',
        title='начальное положение центроид \n'
    )
    axs[1].scatter(*centroids.T, s=100, c='#c00', marker='^')
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(anim_speed)
    # ====================== визуализация ======================
    # Вручную прописанный алгоритм к-средних
    for _ in range(10):
        
        cluster = []
        for point_i in data.values:
            lines = []
            for centr_k in centroids:
                # вычисление расстояния от каждой точки до каждой центроиды
                lines.append(euclid_line(point_i, centr_k))
            # выбор минимального из вычисленных расстояний
            cluster.append(array(lines).argmin())
        
        # ====================== визуализация ======================
        axs[1].clear()
        axs[1].set(
            xlabel='x1', 
            ylabel='x2',
            title='вычисление расстояний до центроид \n и выбор минимального расстояния'
        )
        
        # Назначение цвета для каждого кластера
        for i, point_cluster in enumerate(cluster):
            axs[1].scatter(*data.loc[i], color=colors[point_cluster])
        axs[1].scatter(*centroids.T, s=100, c=colors[:n], marker='^')
        plt.draw()
        plt.gcf().canvas.flush_events()
        sleep(anim_speed*2)
        # ====================== визуализация ======================
        # Нахождение центров каждого кластера
        centroids = []
        for k in range(n):
            # Накладываем маску, чтобы вычленить значения, которые относятся к тому или иному кластеру 
            x1_k = data.loc[array(cluster) == k]['x1']
            x2_k = data.loc[array(cluster) == k]['x2']
            centroids.append((x1_k.mean(), x2_k.mean()))
        centroids = array(centroids)
        
        # ====================== визуализация ======================
        axs[1].set(
            xlabel='x1', 
            ylabel='x2',
            title='вычисление центров полученных кластеров \n'
        )
        axs[1].scatter(*centroids.T, s=200, c=colors[:n], marker='2')
        plt.draw()
        plt.gcf().canvas.flush_events()
        sleep(anim_speed*2)
        
        axs[1].clear()
        axs[1].set(
            xlabel='x1', 
            ylabel='x2',
            title='новое положение центроид \n'
        )
        for i, point_cluster in enumerate(cluster):
            axs[1].scatter(*data.loc[i], color=colors[point_cluster])
        axs[1].scatter(*centroids.T, s=100, c='#c00', marker='^')
        plt.draw()
        plt.gcf().canvas.flush_events()
        sleep(anim_speed/2)
        
        axs[1].clear()
        axs[1].set(
            xlabel='x1', 
            ylabel='x2',
            title='новое положение центроид \n'
        )
        axs[1].scatter(*data.values.T)
        axs[1].scatter(*centroids.T, s=100, c='#c00', marker='^')
        plt.draw()
        plt.gcf().canvas.flush_events()
        sleep(anim_speed/2)
        # ====================== визуализация ======================
    
    
    # ====================== визуализация ======================
    axs[1].clear()
    axs[1].set(
        xlabel='x1', 
        ylabel='x2',
        title='итоговое расположение центроид и кластеров \n'
    )
    for i, point_cluster in enumerate(cluster):
        axs[1].scatter(*data.loc[i], color=colors[point_cluster])
    axs[1].scatter(*centroids.T, s=100, c=colors[:n], marker='^')
    plt.draw()
    plt.gcf().canvas.flush_events()
    # ====================== визуализация ======================
    # Вычисление суммы внутрикластерных расстояний 
    total = 0
    for k in range(n):
        data_k = data.loc[array(cluster) == k]
        for p1, p2 in combinations(data_k.values, 2):
            total += euclid_line(p1, p2)
    wcss.append(total)
    wcss_x = range(2, 2+len(wcss)) 
    
    # Разность WCSS попарно 
    wcss_diff = array([
        n1 - n2
        for n1, n2 in pairwise(wcss)
    ])
    wcss_diff_x = range(2, 2+len(wcss_diff))
    
    # ====================== визуализация ======================
    axs[0].clear()
    axs[0].set(
        xlabel='количество кластеров', 
        ylabel='WCSS',
        title='сумма внутрикластерных расстояний (WCSS) \n'
    )
    axs[0].set_xticks(wcss_x)
    axs[0].plot(wcss_x, wcss, '.-r', lw=2, ms=15)
    axs[0].plot(wcss_diff_x, wcss_diff, 's-c', lw=3, ms=12)
    plt.draw()
    plt.gcf().canvas.flush_events()
    sleep(anim_speed*2)
    # ====================== визуализация ======================
    
    # data = DataFrame(data.to_dict() | {'cluster': {i: c for i, c in enumerate(cluster)}})


plt.ioff()
plt.show()

