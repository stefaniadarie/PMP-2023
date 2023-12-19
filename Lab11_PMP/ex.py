import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generarea a 500 de date dintr-o mixtură de trei distribuții Gaussiene
n_clusters = 3
n_cluster = [170, 165, 165]  # Numărul de date din fiecare distribuție, sumând la 500
means = [5, 0, -3]  # Mediile fiecărei Gaussiene
std_devs = [2, 1, 1.5]  # Deviațiile standard ale fiecărei Gaussiene

# Generarea datelor
data = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

# Plotting (opțional, pentru vizualizare)
plt.hist(data, bins=30, density=True)
plt.title('Histograma Datelor Generate')
plt.show()

# Calibrarea modelelor de mixtură Gaussiană cu 2, 3 și 4 componente
gmm_2 = GaussianMixture(n_components=2, random_state=0).fit(data.reshape(-1, 1))
gmm_3 = GaussianMixture(n_components=3, random_state=0).fit(data.reshape(-1, 1))
gmm_4 = GaussianMixture(n_components=4, random_state=0).fit(data.reshape(-1, 1))

# Calculul logaritmului verosimilității pentru fiecare model
log_likelihood_2 = gmm_2.score_samples(data.reshape(-1, 1))
log_likelihood_3 = gmm_3.score_samples(data.reshape(-1, 1))
log_likelihood_4 = gmm_4.score_samples(data.reshape(-1, 1))

