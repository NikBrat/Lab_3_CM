import numpy as np
import matplotlib.pyplot as plt


def plotting(x, y, name, option, y2=None, y3=None):
    """Создание графиков"""
    match option:
        case 0:
            fig = plt.figure()
            plt.plot(x, y, color='blue')
            plt.xlabel('t')
            plt.ylabel('g(t)')
            plt.title(f'График исходного сигнала g(t)')
            plt.grid()
        case 1:
            fig = plt.figure()
            plt.plot(x, y, color='red')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title('График искаженного сигнала u(t)')
            plt.grid()
        case 2:
            fig, ax = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')
            fig.suptitle('Графики модулей $\hat{u}$\(\u03BD) и его отфильтрованного сигнала (среза)', size='x-large')
            ax[0].plot(x, np.abs(y), color='red')
            ax[1].plot(x, np.abs(y2), color='green')
            ax[0].set_xlabel('\u03BD')
            ax[0].set_ylabel('$\hat{u}$(\u03BD)')
            ax[1].set_xlabel('\u03BD')
            ax[1].set_ylabel('$\hat{u}$\(\u03BD) (срез)')
            ax[0].grid()
            ax[1].grid()
            ax[0].set_title('График модуля $\hat{u}$\(\u03BD)')
            ax[1].set_title('График модуля отфильтрованного сигнала (среза)')
        case 3:
            fig = plt.figure()
            plt.plot(x, y, color='red', label='Искаженный сигнал')
            plt.plot(x, y2, color='green', label='Отфильтрованный сигнал')
            plt.plot(x, y3, color='blue', label='Исходный сигнал', linestyle='--')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title('Графики искаженного (u(t)), отфильтрованного и исходного сигналов')
            plt.grid()
            plt.legend(loc='best')
    # plt.show()
    fig.savefig(f'{name}'.replace('.', ','), dpi=200)


def dot_product(f, g, mn, mx):
    """Вычисление скалярного произведения"""
    x = np.linspace(mn, mx, 1000)
    dx = x[1] - x[0]
    return np.dot(f, g(x)) * dx


def fourier_transformation(f):
    image = lambda v: dot_product(f, lambda t: np.exp(-1j * 2 * np.pi * v * t), -6, 6)
    return np.vectorize(image, otypes=[np.complex_])


def fourier_transformation_inverse(f):
    """Определение функции для вычисления фурье-образа функции"""
    image = lambda t: dot_product(f, lambda v: np.exp(1j * 2 * np.pi * v * t), -3, 3)
    return np.vectorize(image)


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

bs = [4]
clp_freqs = [450]
c = 1
d = 3
for b in bs:
    received = noisy_signal(interval, original, rnd, b, c, d)
    plotting(interval, received, f'Noisy_{b}_{c}_{d}', 1)
    for clp_freq in clp_freqs:
        frequencies = np.linspace(-3, 3, 1000)
        fourier = fourier_transformation(received)(frequencies)
        fourier_clipped = np.concatenate((fourier[0:clp_freq], np.zeros(100), fourier[-clp_freq-1:-1]))
        plotting(frequencies, fourier, f'Fourier_Image_{b}_{c}_{d}_{frequencies[clp_freq]}', 2, y2=np.abs(fourier_clipped))
        fourier_inverse = fourier_transformation_inverse(fourier_clipped)(interval)
        plotting(interval, received, f'Cleaned_{b}_{c}_{d}_{frequencies[clp_freq]}', 3,  y2=fourier_inverse, y3=original)

