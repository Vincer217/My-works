from matplotlib import pyplot as plt
from numpy import diff, log1p
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller

from pathlib import Path
from sys import path


script_dir = Path(path[0])
passengers = read_csv(
    script_dir / 'passengers.csv',
    index_col='month',
    parse_dates=True
)

p_flat = passengers.values.flatten()

# Дифференцирование (удаление тренда)
p_diff = diff(p_flat)
# Сезонное диференцирование (удаление сезонности)
period = 12
p_s_diff = p_flat[period:] - p_flat[:-period]
# Логарифмирование (стабилизация дисперсии)
p_log = log1p(p_flat)


# Все преобразования и проверка стационарности на каждом шаге 
def to_stationary(timeseries, /, season_period, pvalue_crit=0.05):
    """
    Преобразует временной ряд в стационарный, если это необходимо.
    Проверка происходит с помощью критерия p_value:
    - если значение временного ряда меньше, то ряд стационарный;
    - если значение временного ряда больше, то ряд нестационарный и проводится преобразование.
    
    Преобразования:
        1. Логарифмирование;
        2. Дифференцирование;
        3. Сезонное диференцирование.
        
    Параметры: 
        timeseries: ndarray 
            Временной ряд.
        season_period: int
            Период сезонности.
        pvalue_crit: float, optional (default=0.05)
            Критерий проверки на стационарность.
    
    Возвращает:
        ndarray (Стационарный временной ряд).
    """
    if adfuller(timeseries)[1] > pvalue_crit:
        timeseries = log1p(timeseries)
        if adfuller(timeseries)[1] > pvalue_crit:
            timeseries = diff(timeseries)
            if adfuller(timeseries)[1] > pvalue_crit:
                timeseries = timeseries[season_period:] - timeseries[:-season_period]
    return timeseries


fig = plt.figure(figsize=(9, 12))
axs = fig.subplots(5, 1)

axs[0].plot(p_flat)
axs[0].set_title('исходный ряд')
axs[1].plot(p_diff)
axs[1].set_title('дифференцирование')
axs[2].plot(p_s_diff)
axs[2].set_title('сезонное дифференцирование')
axs[3].plot(p_log)
axs[3].set_title('логарифмирование')
axs[4].plot(to_stationary(p_flat, season_period=12))
axs[4].set_title('все преобразования')

fig.savefig(script_dir / 'passengers_to_stationary.png', dpi=150)

