import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# a. Implementarea metodei grid pentru modelul aruncării monedei
def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    Implementeaza metoda grid pentru problema aruncării monedei.
    Aceasta calculează distribuția a posteriori pentru probabilitatea θ a obținerii unei fețe.

    :param grid_points: Numarul de puncte din grid.
    :param heads: Numarul de 'fețe' observate.
    :param tails: Numarul de 'pile' observate.
    :return: O pereche de array-uri reprezentand grid-ul de valori θ și distribuția a posteriori asociata.
    """
    grid = np.linspace(0, 1, grid_points)  # Crearea unui grid de valori pentru θ între 0 și 1
    prior = np.repeat(5, grid_points)  # Selecția unui prior uniform
    likelihood = stats.binom.pmf(heads, heads + tails, grid)  # Calcularea verosimilității pentru fiecare punct din grid
    posterior = likelihood * prior  # Calcularea posteriorului prin inmulțirea verosimilității cu priorul
    posterior /= posterior.sum()  # Normalizarea posteriorului
    return grid, posterior


# Presupunem ca am aruncat de 13 ori o moneda si am observat trei steme
data = np.repeat([0, 1], (10, 3))  # Reprezentarea datelor experimentale
points = 10  # Numarul de puncte din grid
h = data.sum()  # Calcularea numarului de 'fete'
t = len(data) - h  # Calcularea numarului de 'pile'
grid, posterior = posterior_grid(points, h, t)  # Obținerea distribuției a posteriori

# b. Afișarea graficului pentru distribuția a posteriori
plt.plot(grid, posterior, 'o-')  # Plotarea distribuției a posteriori
plt.title(f'heads = {h}, tails = {t}')  # Adaugarea unui titlu la grafic
plt.yticks([])  # Eliminarea etichetelor pe axa y
plt.xlabel('θ')  # Adaugarea unei etichete pe axa x
plt.show()  # Afișarea graficului