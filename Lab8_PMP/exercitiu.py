import pandas as pd
import pymc3 as pm
import numpy as np

# Încărcați datele din fișierul CSV
data = pd.read_csv("Prices.csv")

# Definiți variabilele independente și dependente
X = data[['Speed', 'HardDrive']]
X['HardDrive'] = np.log(X['HardDrive'])  # Transformarea logaritmică a mărimii hard diskului
y = data['Price']

# Model Bayesian
with pm.Model() as model:
    # Prioare pentru coeficienții liniari și deviația standard a erorii
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta1 = pm.Normal('beta1', mu=0, sd=10)
    beta2 = pm.Normal('beta2', mu=0, sd=10)
    sigma = pm.HalfNormal('sigma', sd=10)

    # Model liniar
    mu = alpha + beta1 * X['Speed'] + beta2 * X['HardDrive']

    # Distribuție aprobabilitate a prețului
    price = pm.Normal('price', mu=mu, sd=sigma, observed=y)

    # Estimare folosind MCMC
    trace = pm.sample(2000, tune=1000, cores=1)  # Poate dura ceva timp

# 2. Estimările de 95% HDI ale parametrilor beta1 și beta2:
beta1_hdi = pm.stats.hdi(trace['beta1'])
beta2_hdi = pm.stats.hdi(trace['beta2'])

# 3. Sunt frecvența procesorului și mărimea hard diskului predictori utili ai prețului de vânzare?
# Pentru a răspunde la această întrebare, puteți verifica dacă intervalele HDI ale coeficienților conțin zero. Dacă nu conțin, atunci acei predictorii sunt considerați utili.
predictors_utili = {
    'Frequency (beta1)': beta1_hdi,
    'HardDrive Size (beta2)': beta2_hdi
}
print(predictors_utili)

# 4. Simularea prețului de vânzare pentru un computer cu 33 MHz și 540 MB hard disk:
new_computer_data = pd.DataFrame({'Speed': [33], 'HardDrive': [np.log(540)]})
with model:
    new_computer_mu = alpha + beta1 * new_computer_data['Speed'] + beta2 * new_computer_data['HardDrive']
    new_computer_price_samples = pm.sample_posterior_predictive(trace, samples=5000)['price']

# Calcularea intervalului de 90% HDI pentru prețul așteptat al noului computer
new_computer_hdi = pm.stats.hdi(new_computer_price_samples, hdi_prob=0.9)
print("Interval de 90% HDI pentru prețul așteptat al noului computer:", new_computer_hdi)

# 5. Simularea prețului de vânzare pentru un computer cu 33 MHz și 540 MB hard disk pentru predicție:
with model:
    predictive_samples = pm.sample_posterior_predictive(trace, samples=5000)

# Calcularea intervalului de predicție de 90% HDI pentru prețul noului computer
predictive_hdi = pm.stats.hdi(predictive_samples['price'], hdi_prob=0.9)
print("Interval de predicție de 90% HDI pentru prețul noului computer:", predictive_hdi)
