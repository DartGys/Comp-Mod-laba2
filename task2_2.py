import numpy as np
import matplotlib.pyplot as plt

# Задані точки на площині
mas_x = np.array([-1.6000, -1.2000, -0.8000, -0.4000, 0, 0.4000, 0.8000, 1.2000, 1.6000, 2.0000]) + 4
mas_y = np.array([4.3200, 3.2800, 2.8800, 3.1200, 4.0000, 5.5200, 7.6800, 10.4800, 13.9200, 18.0000]) - 0.04

# Функція для підгонки квадратного трьохчлена
def quadratic_func(x, A, B, C):
    return A * x ** 2 + B * x + C

# Метод найменших квадратів для підгонки
def least_squares_fit(mas_x, mas_y):
    # Побудова матриці дизайну
    X = np.vstack([mas_x ** 2, mas_x, np.ones_like(mas_x)]).T

    # Виконання методу найменших квадратів
    params = np.linalg.lstsq(X, mas_y, rcond=None)[0]

    return params

# Виконання методу найменших квадратів
params = least_squares_fit(mas_x, mas_y)

A, B, C = params

# Виведення оптимальних значень
print(f"A: {A}")
print(f"B: {B}")
print(f"C: {C}")

# Побудова графіку
plt.scatter(mas_x, mas_y, color="blue", label="Дані")
plt.plot(mas_x, quadratic_func(mas_x, *params), color="red", label=f"y = {A:.4f}x^2 + {B:.4f}x + {C:.4f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Підгонка квадратного трьохчлена методом найменших квадратів")
plt.grid(True)
plt.show()
