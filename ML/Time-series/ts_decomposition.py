from matplotlib import pyplot as plt
from pandas import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from pathlib import Path
from sys import path


script_dir = Path(path[0])
passengers = read_csv(
    script_dir / 'passengers.csv',
    index_col='month',
    parse_dates=True
)
births = read_csv(
    script_dir / 'births.csv',
    index_col='date',
    parse_dates=True
)

# Cкользящее среднее для разных значений окна 
passengers_ma_2 = passengers.rolling(2).mean()
passengers_ma_3 = passengers.rolling(3).mean()
passengers_ma_4 = passengers.rolling(4).mean()
passengers_ma_6 = passengers.rolling(6).mean()
passengers_ma_8 = passengers.rolling(8).mean()
passengers_ma_12 = passengers.rolling(12).mean()


fig = plt.figure(figsize=(15, 12))
axs = fig.subplots(3, 2)

# Графики временного ряда и скользящего среднего с разными окнами 
axs[0][0].plot(passengers)
axs[0][0].plot(passengers_ma_2)
axs[0][0].set_title('window = 2')

axs[0][1].plot(passengers)
axs[0][1].plot(passengers_ma_3)
axs[0][1].set_title('window = 3')

axs[1][0].plot(passengers)
axs[1][0].plot(passengers_ma_4)
axs[1][0].set_title('window = 4')

axs[1][1].plot(passengers)
axs[1][1].plot(passengers_ma_6)
axs[1][1].set_title('window = 6')

axs[2][0].plot(passengers)
axs[2][0].plot(passengers_ma_8)
axs[2][0].set_title('window = 8')

axs[2][1].plot(passengers)
axs[2][1].plot(passengers_ma_12)
axs[2][1].set_title('window = 12')

# fig.savefig(script_dir / 'passengers_ma.png', dpi=150)


window_width = 12

# Линия тренда 
passengers_ma = passengers.rolling(window_width).mean().shift(-window_width)

fig = plt.figure(figsize=(7, 10))
axs = fig.subplots(3, 1)

axs[0].plot(passengers)
axs[0].plot(passengers_ma)
axs[0].set_xlim(passengers.index[0], passengers.index[-1])
axs[0].set_title('исходный временной ряд и тренд')

# Сезонность. Вычитаем линию тренда из временного ряда
passengers_season = passengers[:-window_width] - passengers_ma[:-window_width]

axs[1].plot(passengers_season)
axs[1].set_xlim(passengers.index[0], passengers.index[-1])
axs[1].set_title('сезонность до группировки и усреднения')

for i in range(0, len(passengers)-window_width):
    # Группировка
    subgroup = passengers_season[i::window_width] # i - это номер строки, window_width - это шаг 
    # Усреднение 
    passengers_season.iloc[i] = subgroup.mean()

axs[2].plot(passengers_season)
axs[2].set_xlim(passengers.index[0], passengers.index[-1])
axs[2].set_title('сезонность после группировки и усреднения')

# fig.savefig(script_dir / 'passengers_season.png', dpi=150)

# Случайные колебания 
passengers_residuals = passengers - passengers_ma - passengers_season

fig = plt.figure(figsize=(9, 12))
axs = fig.subplots(4, 1)

axs[0].plot(passengers)
axs[0].set_xlim(passengers.index[0], passengers.index[-1])
axs[0].set_title('исходный временной ряд')

axs[1].plot(passengers_ma)
axs[1].set_xlim(passengers.index[0], passengers.index[-1])
axs[1].set_title('тренд')

axs[2].plot(passengers_season)
axs[2].set_xlim(passengers.index[0], passengers.index[-1])
axs[2].set_title('сезонность')

axs[3].scatter(passengers_residuals.index, passengers_residuals.values)
axs[3].set_xlim(passengers.index[0], passengers.index[-1])
axs[3].set_title('остатки')

# fig.savefig(script_dir / 'passengers_decomposition.png', dpi=150)

# Разложение временного ряда рождаемости 
births_decomposition = seasonal_decompose(births)

fig = births_decomposition.plot()
fig.set_size_inches(9, 12)
fig.set_layout_engine('constrained')
# fig.savefig(script_dir / 'births_decomposition.png', dpi=150)

# Dickey-Fuller - тест для проверки на стационарность
passengers_df_test = adfuller(passengers)
births_df_test = adfuller(births)

# Статистический критерий - p-value. Проверка на стационарность 
passengers_stationary = passengers_df_test[1] <= 0.05
births_stationary = births_df_test[1] <= 0.05

print(
    f'для passengers: p-value = {passengers_df_test[1]:.3f} — '
    f'{"стационарный" if passengers_stationary else "нестационарный"}',
    f'для births: p-value = {births_df_test[1]:.3f} — '
    f'{"стационарный" if births_stationary else "нестационарный"}',
    sep='\n'
)

# для passengers: p-value = 0.992 — нестационарный
# для births: p-value = 0.000 — стационарный