import numpy as np
import matplotlib.pyplot as plt


def plotting(x, y, name):
    """Создание графиков"""
    fig = plt.figure(figsize=(8.0, 6.0))
    plt.plot(x, y, color='red')
    plt.xlabel('t')
    plt.ylabel('g(t)')
    plt.title('График функции g(t)')
    plt.grid()
    plt.show()
    # fig.savefig(f'{name}', dpi=200)


def fourier_transformation(f):
    """Определение функции для вычисления фурье-образа функции"""
    image = np.fft.fftshift(np.fft.fft(f))
    return image


def fourier_transformation_inverse(f):
    """Определение функции для вычисления фурье-образа функции"""
    inverse_function = np.fft.ifft(np.fft.ifftshift(f))
    return inverse_function


def initial_function(t):
    """
    Initial function

    g(t) = Q if t1 <= t <= t2 || 0

    Q = 3
    t1, t2 = -2, 5
    """
    if -2 <= t <= 5:
        return 3
    else:
        return 0


def noisy_signal(t: np.ndarray, g, rndm, b, c, d):
    signal = g + b*(rndm - 0.5) + c*np.sin(d*t)
    return signal


initial_function = np.vectorize(initial_function)
interval = np.linspace(-6, 6, 1000)
original = initial_function(interval)

noisy_signal = np.vectorize(noisy_signal)
rng = np.random.default_rng(27)
rnd = rng.random((interval.size,))
received = noisy_signal(interval, original, rnd, b=1, c=0, d=0)
plotting(interval, received, 'Noisy')
