import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pymc3 as pm
import numpy as np

# Încărcați setul de date într-un Pandas DataFrame
df = pd.read_csv('auto-mpg.csv')

# Trasați un grafic de dispersie pentru a vizualiza relația dintre CP și mpg
sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title('Relația dintre Cai putere și Mile pe galon')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()

# Definiți modelul în PyMC folosind CP ca variabilă independentă și mpg ca variabilă dependentă
with pm.Model() as mpg_model:
    # Specificați distribuția prioră pentru coeficientul de regresie și intercepție
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Definiți distribuția așteptată a variabilei dependente (mpg)
    mu = alpha + beta * df['horsepower']

    # Specificați distribuția așteptată a observațiilor (mpg)
    mpg = pm.Normal('mpg', mu=mu, sd=10, observed=df['mpg'])

# Determinați dreapta de regresie care se potrivește cel mai bine datelor
with mpg_model:
    trace = pm.sample(1000, tune=1000)

# Afișați rezultatele
print(pm.summary(trace).round(2))

# Adăugați graficului de la punctul a regiunea 95%HDI pentru distribuția predictivă a posteriori
sns.scatterplot(x='horsepower', y='mpg', data=df)
pm.plot_posterior_predictive_glm(trace, samples=100,
                                 eval=np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100), color='blue',
                                 alpha=0.1)
plt.title('Relația dintre Cai putere și Mile pe galon cu HDI')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()
