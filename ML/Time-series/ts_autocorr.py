from matplotlib import pyplot as plt
from numpy import corrcoef
from pandas import read_csv
from statsmodels.graphics.tsaplots import plot_acf

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

# shifts = [0, 1, 2, 3, 4, 6, 12]
shifts = [0, 4, 6, 12]
# colors = ['#7277fe', '#6c92f0', '#66ade1', '#5fc9d3', '#59e4c5', '#acf2e2', '#d6f9f1']
colors = ['#7277fe', '#66ade1', '#59e4c5', '#d6f9f1']

fig = plt.figure(figsize=(10, 5))
axs = fig.subplots()

# Сдвиги на графике
for sh, color in zip(shifts, colors):
    axs.plot(passengers.shift(sh), c=color)

fig.show()


fig = plt.figure(figsize=(9, 7))
axs = fig.subplots()

axs.vlines(0, 0, 1, colors='black')
axs.scatter(0, 1, c=colors[0])

# Корреляция между данными с разными сдвигами 
for shift in range(1, 31):
    # Коэффициент корреляции между значениями с разным сдвигом
    coef = corrcoef(passengers.values.flatten()[:-shift], passengers.values.flatten()[shift:])[0, 1]
    axs.vlines(shift, 0, coef, colors='black')
    axs.scatter(shift, coef, c=colors[0])

axs.set_ylim(-1.1, 1.1)

fig.show()


fig = plt.figure(figsize=(9, 7))
axs = fig.subplots()

axs.vlines(0, 0, 1, colors='black')
axs.scatter(0, 1, c=colors[0])

# Корреляция между данными с разными сдвигами
for shift in range(1, 31):
    coef = corrcoef(births.values.flatten()[:-shift], births.values.flatten()[shift:])[0, 1]
    print(f'{shift = }, {coef = :.3f}')
    axs.vlines(shift, 0, coef, colors='black')
    axs.scatter(shift, coef, c=colors[0])

axs.set_ylim(-1.1, 1.1)

fig.show()


fig = plot_acf(passengers, lags=range(31))
fig.show()


fig = plot_acf(births, lags=range(31))
fig.show()


