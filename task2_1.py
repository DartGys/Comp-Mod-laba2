import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Задані точки на площині
mas_x = np.array([-1.6000, -1.2000, -0.8000, -0.4000, 0, 0.4000, 0.8000, 1.2000, 1.6000, 2.0000]) + 4
mas_y = np.array([0.1630, 0.3629, 0.8076, 1.7973, 4.0000, 8.9022, 19.8121, 44.0927, 98.1301, 218.3926]) + 0.04

# Лінієаризована функція
def linear_func(x, a, b):
    return a * x + b

# Пошук оптимальних параметрів A, B, та C
best_B = None
best_params = None
min_error = float("inf")

for B in np.linspace(0, 10, 500):
    Y = np.log(mas_y + B)
    params, _ = curve_fit(linear_func, mas_x, Y)
    error = np.sum((Y - linear_func(mas_x, *params)) ** 2)
    if error < min_error:
        min_error = error
        best_B = B
        best_params = params

A, lnC = best_params
C = np.exp(lnC)

# Виведення оптимальних значень
print("Optimal values:")
print(f"A: {A}")
print(f"B: {best_B}")
print(f"C: {C}")

# Побудова графіків
plt.scatter(mas_x, mas_y, color="blue", label="Задані точки")
plt.plot(mas_x, C * np.exp(A * mas_x) - best_B, color="red", label=f'C * e^(A * x) - B з A={A:.4f}, C={C:.4f}, B={best_B:.4f}')
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Підгонка кривих методом найменших квадратів")
plt.grid(True)
plt.show()
