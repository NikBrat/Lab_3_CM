import numpy as np
import matplotlib.pyplot as plt


def dot_product(f, g):
    """Вычисление скалярного произведения"""
    x = np.linspace(-6, 6, 1000)
    dx = x[1] - x[0]
    return np.dot(f, g(x)) * dx


def fourier_transformation(x):
    """Определение функции для вычисления фурье-образа функции"""
    image = lambda v: dot_product(x, lambda t: np.exp(-1j * 2 * np.pi * v * t))
    return np.vectorize(image, otypes=[np.complex_])


def fourier_transformation_inverse(x):
    """Определение функции для вычисления фурье-образа функции"""
    inverse_function = lambda t: (x, lambda v: np.exp(1j * 2*np.pi * v * t))
    return np.vectorize(inverse_function, otypes=[np.complex_])


def function(t):
    """
    Initial function

    g(t) = Q if t1 <= t <= t2 || 0

    Q = 3
    t1, t2 = -1, 5
    """
    if 1 <= t <= 5:
        return 3
    else:
        return 0


interval = np.linspace(-6, 6, 1000)
