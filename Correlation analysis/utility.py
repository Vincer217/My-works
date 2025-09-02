from numpy import array
from matplotlib import pyplot as plt
from time import sleep

def coef_corr(X: array, Y: array) -> float:
    """
    Функция, которая вычисляет коэффициент корреляции.
    """
    corr_moment = sum(X * Y / len(X)) - X.mean()*Y.mean()    
    corr_coef = corr_moment / (X.std()*Y.std())
                 
    return corr_coef 
    
def countable_nouns(num: int, nouns: tuple[str, str, str]) -> str:
    """
    Функция возвращает одно слово из трёх, переданных вторым аргументом, которое согласуется с переданным первым аргументом числом.
    """
    last_two_digits = num % 100 
    last_digit = num % 10  

    if (
           11 <= last_two_digits <= 19 
        or last_digit == 0 
        or 5 <= last_digit <= 9
    ):
        return nouns[2]  
    elif 2 <= last_digit <= 4:
        return nouns[1]  
    elif last_digit == 1:
        return nouns[0]
        
def find_max_correlation(X, Y, max_shift=5):
    """
    Функция, которая находит сдвиг с максимальной корреляцией между показателями.
    """
    
    plt.ion()
    
    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots()
    
    # Нормализация
    X_norm = (X - X.mean()) / X.std()
    Y_norm = (Y - Y.mean()) / Y.std()
    
    max_r = -1
    best_shift = 0
    results = []
    
 
    for shift in range(0, max_shift+1):
        if shift == 0:
            x = X_norm[:len(Y_norm)]
            y = Y_norm
        else:
            x = X_norm[shift:len(Y_norm)+shift]
            y = Y_norm[:len(X_norm)-shift]
        
        r = coef_corr(x, y)
        
        results.append({
            'shift': shift,
            'X_values': x,
            'Y_values': y,
            'correlation': r
        })

        if r > max_r:
            max_r = r
            best_shift = shift
            
        axs.clear()
        axs.set(title=f'coef_corr={r}')
        axs.plot(x, y, '^-', ms=12)
        plt.draw()
        fig.canvas.flush_events()
        
        sleep(2)
        
    plt.ioff()
    plt.show()
    
    return results, best_shift, max_r        