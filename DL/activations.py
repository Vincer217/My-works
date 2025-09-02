from matplotlib import pyplot as plt
from numpy import (
    array,
    exp,
    linspace,
)


def step(x: float, cutoff: float = 0) -> float:
    """Ступенчатая функция (Хэвисайда)."""
    if x < cutoff:
        return 0
    else:
        return 1


def linear(x: float, slope: float = 1, shift: float = 0) -> float:
    """Линейная функция."""
    return x*slope - shift


def relu(x: float, slope: float = 1, cutoff: float = 0) -> float:
    """Кусочно-линейная функция."""
    if x < cutoff:
        return 0
    else:
        return x*slope - cutoff


def lrelu(
            x: float, 
            slope_right: float = 1, 
            slope_left: float = 0.1, 
            cutoff: float = 0
    ) -> float:
        """Кусочно-линейная функция c "утечкой"."""
        if x < cutoff:
            return (x - cutoff)*slope_left
        else:
            return x*slope_right - cutoff


def prelu(
            x: float,
            left_cutoff: float = -1,
            right_cutoff: float = 1,
    ) -> float:
        """Усечённая (параметризованная) кусочно-линейная функция с нормализацией.
        
        :param x: входное значение
        :param left_cutoff: левый (нижний) порог
        :param right_cutoff: правый (верхний) порог
        """
        if x < left_cutoff:
            return 0
        elif right_cutoff < x:
            return 1
        else:
            slope = 1 / (right_cutoff - left_cutoff)
            shift = left_cutoff * slope
            return x * slope - shift


def sigmoid(x: float, a: float = 1, b: float = 1):
    """Сигмоидальная логистическая функция. 
    
    :param a: угол наклона
    :param b: смещение
    """
    return 1 / (1 + exp(-a*x + b))


def tanh(x, a=1, b=0):
    """Функция гиперболического тангенса."""
    return 2 / (1 + exp(-2*a*x + b)) - 1



if __name__ == '__main__':
    
    x = linspace(-10, 10, 101)
    
    f_relu = [relu(n) for n in x]
    f_lrelu = [lrelu(n, cutoff=3) for n in x]
    f_prelu = [prelu(n, -3, 5) for n in x]
    f_sigm1 = [sigmoid(n, a=0.5) for n in x]
    f_sigm2 = [sigmoid(n, a=2) for n in x]
    f_tanh = [tanh(n, a=0.8) for n in x]
    
    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(3, 2)
    
    axs[0][0].plot(x, f_relu)
    axs[1][0].plot(x, f_lrelu)
    axs[2][0].plot(x, f_prelu)
    axs[0][1].plot(x, f_sigm1)
    axs[1][1].plot(x, f_sigm2)
    axs[2][1].plot(x, f_tanh)
    
    fig.show()


