import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

# Generăm date de antrenare folosind o distribuție normală cu parametrii a priori
np.random.seed(42)  # Pentru reproducibilitate
alfa_prior = 2.0    # Parametrul alfa pentru distribuția a priori
miu_prior = 10.0    # Parametrul miu pentru distribuția a priori
timpi_medii_asteptare = np.random.normal(loc=miu_prior, scale=alfa_prior, size=100)

# Definim modelul Bayesian în PyMC
def run_model():
    with pm.Model() as model:
        # Distribuții a priori pentru alfa și miu
        alfa = pm.Uniform('alfa', lower=0, upper=10)  # Distribuție uniformă pentru alfa
        miu = pm.Normal('miu', mu=0, sigma=20)       # Distribuție normală pentru miu cu deviație standard mare

        # Distribuție a posteriori pentru miu, bazată pe datele observate
        observatii = pm.Normal('observatii', mu=miu, sigma=alfa, observed=timpi_medii_asteptare)

        # Eșantionarea din distribuția a posteriori
        trace = pm.sample(1000, tune=1000, random_seed=42)

    # Distribuția a posteriori pentru miu
    posterior_miu = trace['miu']

    # Vizualizare grafică a distribuției a posteriori pentru miu
    plt.figure(figsize=(10, 6))
    plt.hist(posterior_miu, bins=30, density=True, alpha=0.5, color='blue', label='Distribuția a posteriori')
    plt.title('Distribuția a posteriori pentru miu')
    plt.xlabel('Timpul Mediu de Așteptare la Coadă')
    plt.ylabel('Densitate')
    plt.legend()
    plt.show()

run_model()

# Justificarea acestor alegeri se bazează pe principiul că alegerea distribuțiilor
# a priori trebuie să fie cât mai puțin influențată de datele observate și să reflecte
# incertitudinea noastră în legătură cu parametrii modelului înainte de a vedea datele.
# În acest caz, distribuțiile a priori alese sunt considerate neinformative sau puțin informative,
# permițând datelor să aibă un impact semnificativ asupra distribuțiilor a posterioare estimate.