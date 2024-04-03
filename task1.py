import numpy as np
import matplotlib.pyplot as plt


def plotting(x, y, name, option, y2=None, y3=None):
    """Создание графиков"""
    match option:
        case 0:
            # Исходная функция
            fig = plt.figure()
            plt.plot(x, y, color='blue')
            plt.xlabel('t')
            plt.ylabel('g(t)')
            plt.title(f'График исходного сигнала g(t)')
            plt.grid()
        case 1:
            # Полученный сигнал
            fig = plt.figure()
            plt.plot(x, y, color='red')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title('График искаженного сигнала u(t)')
            plt.grid()
        case 2:
            # Модули Фурье-образов сигнала и его среза
            fig, ax = plt.subplots(1, 2, figsize=(10, 6), layout='constrained')
            fig.suptitle('Графики модулей $\hat{u}$(\u03BD) и его отфильтрованного сигнала (среза)', size='x-large')
            ax[0].plot(x, np.abs(y), color='red')
            ax[1].plot(x, np.abs(y2), color='green')
            ax[0].set_xlabel('\u03BD')
            ax[0].set_ylabel('$\hat{u}$(\u03BD)')
            ax[1].set_xlabel('\u03BD')
            ax[1].set_ylabel('$\hat{u}$(\u03BD) (срез)')
            ax[1].set_ylim(top=np.max(y))
            ax[0].grid()
            ax[1].grid()
            ax[0].set_title('График модуля $\hat{u}$(\u03BD)')
            ax[1].set_title('График модуля отфильтрованного сигнала')
        case 3:
            # Полученный, очищенный и исходные сигналы
            fig = plt.figure()
            plt.plot(x, y, color='red', label='Искаженный сигнал')
            plt.plot(x, y2, color='green', label='Отфильтрованный сигнал')
            plt.plot(x, y3, color='blue', label='Исходный сигнал', linestyle='--')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title('Графики искаженного (u(t)), отфильтрованного и исходного сигналов')
            plt.grid()
            plt.legend(loc='best')
        case 4:
            # Модули Фурье-образов исходной и очищенного сигналов
            fig = plt.figure()
            plt.plot(x, y2, color='green', label='|$\hat{g}$(\u03BD)|')
            plt.plot(x, y, label='|$\hat{f}$(\u03BD)|(clipped)', linestyle='--', color='blue')
            plt.xlabel('\u03BD')
            plt.ylabel('|$\hat{f}$(\u03BD)|')
            plt.title('Графики модуля образа $\hat{g}$(\u03BD) и отфильтрованного сигнала')
            plt.grid()
            plt.legend(loc='best')
    # plt.show()
    fig.savefig(f'{name}'.replace('.', ','), dpi=200)
    plt.close(fig)


def dot_product(f, g, mn, mx):
    """Вычисление скалярного произведения"""
    x = np.linspace(mn, mx, 1000)
    dx = x[1] - x[0]
    return np.dot(f, g(x)) * dx


def fourier_transformation(f):
    """Определение функции для фурье-преобразования"""
    image = lambda v: dot_product(f, lambda t: np.exp(-1j * 2 * np.pi * v * t), -6, 6)
    return np.vectorize(image, otypes=[np.complex_])


def fourier_transformation_inverse(f):
    """Определение функции для обратного фурье-преобразования"""
    image = lambda t: dot_product(f, lambda v: np.exp(1j * 2 * np.pi * v * t), -3, 3)
    return np.vectorize(image)


def initial_function(t):
    """
    Initial function.

    g(t) = Q if t1 <= t <= t2 || 0

    Q = 3
    t1, t2 = -2, 5
    """
    if -2 <= t <= 5:
        return 3
    else:
        return 0


def noisy_signal(t: np.ndarray, g, rndm, b, c, d):
    """Функция для искажения исходного сигнала"""
    signal = g + b*(rndm - 0.5) + c*np.sin(d*t)
    return signal


initial_function = np.vectorize(initial_function)
interval = np.linspace(-6, 6, 1000)
frequencies = np.linspace(-3, 3, 1000)
original = initial_function(interval)
original_fourier = fourier_transformation(original)(frequencies)

noisy_signal = np.vectorize(noisy_signal)
rng = np.random.default_rng(27)
rnd = rng.random((interval.size,))

bs = [0]
clp_freqs = [200, 300, 400]
cs = [4]
ds = [4]
cf = -0.805
cl = -0.55
for b in bs:
    for c in cs:
        for d in ds:
            received = noisy_signal(interval, original, rnd, b, c, d)
            plotting(interval, received, f'Noisy_{b}_{c}_{d}', 1)
            for clp_freq in clp_freqs:
                fourier = fourier_transformation(received)(frequencies)
                fourier_clipped = np.concatenate((np.zeros(clp_freq), fourier[clp_freq:-clp_freq], np.zeros(clp_freq)))
                bg = np.where(frequencies <= cf)[0][-1]
                ed = np.where(frequencies >= cl)[0][0]
                fourier_clipped[bg:ed] = 0.0
                fourier_clipped[-ed:-bg] = 0.0
                plotting(frequencies, np.abs(fourier),f'Fourier_Image_{b}_{c}_{d}_{frequencies[clp_freq]}_{cf}:{cl}', 2, y2=np.abs(fourier_clipped))
                plotting(frequencies, np.abs(fourier_clipped), f'Fourier_Image_Comparison_{b}_{c}_{d}_{frequencies[clp_freq]}_{cf}:{cl}', 4, y2=np.abs(original_fourier))
                fourier_inverse = fourier_transformation_inverse(fourier_clipped)(interval)
                plotting(interval, received, f'Cleaned_{b}_{c}_{d}_{frequencies[clp_freq]}__{cf}:{cl}', 3,  y2=fourier_inverse, y3=original)
