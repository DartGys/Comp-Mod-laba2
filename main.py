import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані
mas_x = np.array([1, 2, 3, 4, 5, 6])  # Значення аргументів
mas_y = 0.5 * np.sqrt(mas_x) + 2 * np.cos(mas_x)  # Значення функції f(x)

# Функція для побудови графіка
def plot_function(x, y, label, title):
    plt.plot(x, y, 'o', label=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    
    plt.grid(True)

# Функція для обчислення коефіцієнтів прямої лінії y = Ax + B
def linear_fit(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# Функція для обчислення коефіцієнта A та степеня M степеневої функції y = Ax^M
def power_fit(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    A, log_M = np.polyfit(log_x, log_y, 1)
    M = np.exp(log_M)
    return A, M

# Функція для обчислення коефіцієнтів A та C показникової функції y = Ce^(Ax)
def exponential_fit(x, y):
    A, C = np.polyfit(x, np.log(y), 1)
    C = np.exp(C)
    return A, C

# Функція для обчислення коефіцієнтів поліномів y = Ax^2 + Bx + C
def polynomial_fit(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    return coeffs

# Виконуємо підгонку для кожного типу функції
# Пряма лінія y = Ax + B
m, c = linear_fit(mas_x, mas_y)
linear_y = m * mas_x + c
plot_function(mas_x, mas_y, 'Original Data', 'Linear Fit')
plt.plot(mas_x, linear_y, label=f'Linear Fit: y = {m:.2f}x + {c:.2f}')

# Степенева функція y = Ax^M
A, M = power_fit(mas_x, mas_y)
power_y = A * mas_x ** M
plot_function(mas_x, mas_y, 'Original Data', 'Power Fit')
plt.plot(mas_x, power_y, label=f'Power Fit: y = {A:.2f}x^{M:.2f}')

# Показникова функція y = Ce^(Ax)
A, C = exponential_fit(mas_x, mas_y)
exponential_y = C * np.exp(A * mas_x)
plot_function(mas_x, mas_y, 'Original Data', 'Exponential Fit')
plt.plot(mas_x, exponential_y, label=f'Exponential Fit: y = {C:.2f}e^{A:.2f}x')

# Поліноми
degree = 2
coeffs = polynomial_fit(mas_x, mas_y, degree)
poly_y = np.polyval(coeffs, mas_x)
plot_function(mas_x, mas_y, 'Original Data', 'Polynomial Fit (Degree 2)')
plt.plot(mas_x, poly_y, label=f'Polynomial Fit (Degree 2): y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}')

degree = 3
coeffs = polynomial_fit(mas_x, mas_y, degree)
poly_y = np.polyval(coeffs, mas_x)
plot_function(mas_x, mas_y, 'Original Data', 'Polynomial Fit (Degree 3)')
plt.plot(mas_x, poly_y, label=f'Polynomial Fit (Degree 3): y = {coeffs[0]:.2f}x^3 + {coeffs[1]:.2f}x^2 + {coeffs[2]:.2f}x + {coeffs[3]:.2f}')

degree = 4
coeffs = polynomial_fit(mas_x, mas_y, degree)
poly_y = np.polyval(coeffs, mas_x)
plot_function(mas_x, mas_y, 'Original Data', 'Polynomial Fit (Degree 4)')
plt.plot(mas_x, poly_y, label=f'Polynomial Fit (Degree 4): y = {coeffs[0]:.2f}x^4 + {coeffs[1]:.2f}x^3 + {coeffs[2]:.2f}x^2 + {coeffs[3]:.2f}x + {coeffs[4]:.2f}')

# Виводимо графіки
plt.legend()
plt.show()
