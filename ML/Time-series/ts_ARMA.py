from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from ts_to_stationary import p_flat, to_stationary

fig = plt.figure(figsize=(9, 12))
axs = fig.subplots(2, 1)

# Точки, которые находятся вне диапозона границ значимости, являются достоверными (значимыми). Первый способ: считается количество точек, но только те, которые значительно удалены от границы и это число является параметром к модели. Второй способ: посчитать точки, которые приближены к 0 и значительно удалены от границы.   
plot_acf(to_stationary(p_flat, 12), ax=axs[0])
plot_pacf(to_stationary(p_flat, 12), ax=axs[1])

fig.show()

# p = 1, d = 0, q = 1
model1 = ARIMA(to_stationary(p_flat, 12), order=(1, 0, 1)) # второй способ 
model2 = ARIMA(to_stationary(p_flat, 12), order=(2, 0, 2)) # первый способ

model_results1 = model1.fit()
p_predicted1 = model_results1.predict(end=140)

model_results2 = model2.fit()
p_predicted2 = model_results2.predict(end=140)


fig = plt.figure(figsize=(9, 12))
axs = fig.subplots(2, 1)

axs[0].plot(to_stationary(p_flat, 12))
axs[0].plot(p_predicted1)

axs[1].plot(to_stationary(p_flat, 12))
axs[1].plot(p_predicted2)

fig.show()



