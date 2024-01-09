import numpy as np
import matplotlib.pyplot as plt

# Generez o grilă de valori posibile pentru parametrul nostru (probabilitatea p)
grid_points = 1000
grid = np.linspace(0, 1, grid_points)

# Exemplu de date observate: număr de succese și eșecuri
successes = 10
failures = 5

# Diferite distribuții a priori
prior_uniform = np.ones(grid_points)  # Distribuție uniformă
prior_biased = (grid <= 0.5).astype(int)  # Favorizează valori <= 0.5
prior_centered = abs(grid - 0.5)  # Favorizează valori apropiate de 0.5

# Calculul verosimilității (likelihood) și posterior
def compute_posterior(prior):
    likelihood = grid**successes * (1 - grid)**failures
    unnormalized_posterior = likelihood * prior
    posterior = unnormalized_posterior / unnormalized_posterior.sum()
    return posterior

# Calculez posterior pentru fiecare prior
plt.figure(figsize=(12, 6))
for prior in [prior_uniform, prior_biased, prior_centered]:
    posterior = compute_posterior(prior)
    plt.plot(grid, posterior, label=str(prior[:3]))

plt.title("Posterior distributions with different priors")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.legend()
plt.show()