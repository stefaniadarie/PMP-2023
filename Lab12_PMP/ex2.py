import numpy as np
import matplotlib.pyplot as plt

# Funcția pentru estimarea lui π folosind metoda Monte Carlo
def estimate_pi(N):
    points = np.random.uniform(-1, 1, size=(N, 2))
    inside_circle = np.sum(points**2, axis=1) <= 1
    pi_estimate = 4 * np.mean(inside_circle)
    return pi_estimate

# Valorile pentru N și numărul de încercări
N_values = [100, 1000, 10000]
trials = 100

# erorile pentru fiecare N
errors = {}

# Calculez estimările și eroarea pentru fiecare N
for N in N_values:
    estimates = np.array([estimate_pi(N) for _ in range(trials)])
    error = np.abs(estimates - np.pi)
    errors[N] = error
    mean_error = np.mean(error)
    std_error = np.std(error)
    print(f"N={N}, Mean Error={mean_error:.5f}, Standard Deviation={std_error:.5f}")

# Vizualizarea erorilor
plt.figure(figsize=(10, 6))
for N in N_values:
    plt.errorbar(N, np.mean(errors[N]), yerr=np.std(errors[N]), fmt='o', label=f'N = {N}')

plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare medie')
plt.title('Eroarea medie și deviația standard a estimării lui π în funcție de N')
plt.legend()
plt.xscale('log')
plt.show()