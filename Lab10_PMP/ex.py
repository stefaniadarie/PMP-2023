import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pymc3 as pm
import arviz as az

#  1a, 1b
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 3 * X**5 - 5 * X**4 + X**3 - X**2 + 5 * X + np.random.normal(0, 10, (100, 1))

poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = model.predict(poly.transform(X_plot))
plt.scatter(X, y, label='Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Regression')
plt.legend()
plt.show()

# 2
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_plot = model.predict(poly.transform(X_plot))
plt.scatter(X, y, label='Data')
plt.plot(X_plot, y_plot, color='red', label='Polynomial Regression')
plt.legend()
plt.show()

# 3
X_cubic = np.random.rand(500, 1) * 10
y_cubic = 3 * X_cubic**3 - 5 * X_cubic**2 + X_cubic + np.random.normal(0, 10, (500, 1))

with pm.Model() as model_cubic:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=3)
    sigma = pm.HalfNormal('sigma', sd=1)

    mu = alpha + beta[0]*X_cubic + beta[1]*X_cubic**2 + beta[2]*X_cubic**3

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y_cubic)

    trace = pm.sample(1000, return_inferencedata=True)

waic = az.waic(trace)
loo = az.loo(trace)

print("WAIC:", waic)
print("LOO:", loo)

az.plot_trace(trace)
plt.show()