import librosa as lr
from scipy.io.wavfile import write
import numpy as np
import matplotlib.pyplot as plt


def dot_product(f, g, mn, mx, num):
    """Вычисление скалярного произведения"""
    x = np.linspace(mn, mx, num)
    dx = x[1] - x[0]
    return np.dot(f, g(x)) * dx


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
            plt.xlabel('\u03C9')
            plt.ylabel('$\hat{u}$(\u03C9)')
            plt.title('Графики модуля Фурье-образа исходной записи')
            plt.grid()
        case 2:
            # Модули Фурье-образов сигнала и его среза
            fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
            fig.suptitle('Графики модулей Фурье-образов исходного и отфильтрованного сигнала', size='x-large')
            ax[0].plot(x, np.abs(y), color='red')
            ax[1].plot(x, np.abs(y2), color='green')
            ax[0].set_xlabel('\u03C9')
            ax[0].set_ylabel('$\hat{u}$(\u03C9)')
            ax[1].set_xlabel('\u03C9')
            ax[1].set_ylim(top=np.max(y))
            ax[1].set_ylabel('$\hat{u}$(\u03C9)(fitered)')
            ax[0].grid()
            ax[1].grid()
            ax[0].set_title('График модуля исходного сигнала')
            ax[1].set_title('График модуля отфильтрованного сигнала')
        case 3:
            # Полученный, очищенный и исходные сигналы
            fig = plt.figure()
            plt.plot(x, y, color='red', label='Исходный сигнал')
            plt.plot(x, y2, color='green', label='Отфильтрованный сигнал')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title('Графики исходного и отфильтрованного сигналов')
            plt.grid()
            plt.legend(loc='best')
        case 4:
            # Модули Фурье-образов исходной и очищенного сигналов
            fig = plt.figure()
            plt.plot(x, y, color='red', label='Исходный сигнал')
            plt.plot(x, y2, label='Отфильтрованный сигнал', linestyle='--', color='green')
            plt.xlabel('\u03C9')
            plt.ylabel('|$\hat{u}$(\u03C9)|')
            plt.title('Графики модуля Фурье-образа исходного и отфильтрованного сигналов')
            plt.grid()
            plt.legend(loc='best')
    plt.show()
    fig.savefig(f'{name}'.replace('.', ','), dpi=200)
    plt.close(fig)


def displaying(audio, freq, tim, name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tim, audio, color='red')
    fig.suptitle(f'График исходного сигнала')
    ax.set_xlabel('Время, t')
    ax.grid()
    plt.show()
    fig.savefig(f'{name}_Звуковая волна'.replace('.', ','), dpi=200)
    plt.close(fig)


sound, rate = lr.load('MUHA.wav')
time = np.linspace(0, stop=(1/rate)*sound.shape[-1], num=124152)
displaying(sound, rate, time, 'Source')


fourier_original = (np.fft.fft(sound))
v = np.fft.fftfreq(len(sound), 1/rate)

fourier_cleaned = np.copy(fourier_original)

fourier_cleaned[np.where(np.abs(v) <= 300)] = 0.0
fourier_cleaned[np.where(np.abs(v) >= 4500)] = 0.0

cleaned = np.fft.ifft(fourier_cleaned).real.astype(np.float32)

plotting(v, np.abs(fourier_original), 'Image_Orig', 1)
plotting(v, np.abs(fourier_original), 'Images', 2, y2=np.abs(fourier_cleaned))
plotting(v, np.abs(fourier_original), 'Images_Comparison', 4, y2=np.abs(fourier_cleaned))
plotting(time, sound, 'Wave_comparison', 3, y2=cleaned.astype(np.float32))
write('cleaned.wav', rate=int(rate), data=cleaned.astype(np.float32))
