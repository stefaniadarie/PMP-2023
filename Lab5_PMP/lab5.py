import pymc as pm
import numpy as np

# Încărcați datele de trafic
traffic_data = np.genfromtxt('trafic.csv', delimiter=',')

alpha, sigma = 1, 1
beta = [1, 2.5]

size = 100

# Definirea modelului
with pm.Model() as traffic_model:
    # Prior pentru parametrul λ
    lambda_ = pm.Uniform('lambda', lower=0, upper=10)

    # Lambda values for specific intervals
    lambda_values = pm.math.switch(
        pm.math.eq(7, np.arange(20)), lambda_ * 2,
        pm.math.switch(pm.math.eq(8, np.arange(20)), lambda_ * 0.5,
                       pm.math.switch(pm.math.eq(16, np.arange(20)),
                                      lambda_ * 3,
                                      pm.math.switch(pm.math.eq(19, np.arange(20)),
                                                     lambda_ * 0.7, lambda_ * 1))))

    # Observații ale traficului
    traffic_obs = pm.Poisson('traffic', mu=lambda_values, observed=traffic_data)

# Inferență Bayesiană
with traffic_model:
    trace = pm.sample(1000, tune=1000, cores=1)

# Afișarea rezultatelor
pm.summary(trace)