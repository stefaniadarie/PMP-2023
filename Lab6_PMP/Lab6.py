import pymc3 as pm
import arviz as az
import numpy as np

Y_values = [0.5, 10]
theta_values = [0.2, 0.5]

for Y in Y_values:
    for theta in theta_values:
        with pm.Model() as model:
            n = pm.Poisson("n", 10)
            y_obs = np.array([Y] * 1000)  # Generăm o listă de 1000 de valori Y identice
            y_var = pm.Binomial("Y_var", n=n, p=theta, observed=y_obs)
            idata = pm.sample(1000, return_inferencedata=True)  # Generăm distribuția a posteriori
                # = pm.sample(1000, tune=1000, cores=1)
            
            # pred_dists = (pm.sample_prior_predictive(1000, model)["y_obs"],
            #               pm.sample_posterior_predictive(idata, 1000, model)["y_obs"])

            az.plot_posterior(idata)  # Vizualizăm distribuția a posteriori
