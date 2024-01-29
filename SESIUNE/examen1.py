import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# a. Încarcați setul de date
df = pd.read_csv('BostonHousing.csv')
X = df[['crim', 'rm', 'indus']]  # Selecționam coloanele 'crim', 'rm', 'indus' ca variabile independente
y = df['medv']  # 'medv' este variabila dependentă (prețul mediu al caselor)
model = LinearRegression()
model.fit(X, y)  # Construim si antrenam modelul de regresie liniara

# b. Definiti coeficienții
coefficients = model.coef_  # Coeficienții pentru fiecare variabilă independenta
intercept = model.intercept_  # Interceptul modelului
print("Coeficienții modelului:", coefficients)
print("Interceptul modelului:", intercept)

# c. Obtineti estimari de 95% pentru HDI ale parametrilor
n = len(X)  # Numarul de observații
p = X.shape[1]  # Numarul de variabile independente
alpha = 0.05  # Nivelul de semnificație
dof = n - p - 1  # Gradele de libertate
t_crit = np.abs(stats.t.ppf((1 - alpha) / 2, dof))  # Valoarea critică t pentru intervalul de încredere de 95%
y_pred = model.predict(X)  # Predicțiile modelului
residuals = y - y_pred  # Calculam reziduurile
residual_std_error = np.sqrt(np.sum(residuals**2) / dof)  # Eroarea standard a reziduurilor
X_with_intercept = np.c_[np.ones((n, 1)), X]  # Adaugam o coloana de 1 pentru intercept
cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * residual_std_error**2  # Matricea de covarianță
standard_errors = np.sqrt(np.diag(cov_matrix)[1:])  # Erorile standard ale coeficienților
conf_intervals = np.column_stack((coefficients - t_crit * standard_errors,
                                  coefficients + t_crit * standard_errors))  # Calculam intervalele de încredere

print("Intervalul de încredere la 95% pentru coeficienții modelului:", conf_intervals)

# Determinăm care dintre coeficienți are cel mai mare impact pe baza intervalului lor de incredere
# Un interval de încredere mai mic sugerează o estimare mai precisă a coeficientului
coef_impact = np.abs(coefficients) / standard_errors  # Raportul dintre coeficient și eroarea standard
max_impact_index = np.argmax(coef_impact)  # Indexul coeficientului cu cel mai mare impact
most_influential_var = X.columns[max_impact_index]  # Numele variabilei cu cel mai mare impact

print("Variabila cu cel mai mare impact pe rezultat:", most_influential_var)

# Afișam coeficienții, erorile standard și intervalele de încredere
for var, coef, se, ci in zip(X.columns, coefficients, standard_errors, conf_intervals):
    print(f"Variabila: {var}, Coeficient: {coef:.2f}, Eroare Standard: {se:.2f}, Interval de Încredere: {ci}")

# d. Simulați extrageri din distribuția predictivă pentru a găsi un interval de predicție de 50%
num_simulations = 1000  # Numarul de simulații
simulated_medv = np.zeros((num_simulations, len(y)))  # Inițializăm un array pentru a stoca rezultatele simulărilor
for i in range(num_simulations):
    random_residuals = np.random.choice(residuals, size=len(residuals), replace=True)  # Alegem reziduuri la întâmplare
    simulated_medv[i, :] = y_pred + random_residuals  # Adăugam reziduurile alese la predicțiile modelului
lower_bounds = np.percentile(simulated_medv, 25, axis=0)  # Calculam percentila 25
upper_bounds = np.percentile(simulated_medv, 75, axis=0)  # Calculăm percentila 75
mean_lower_bound = np.mean(lower_bounds)  # Media limitelor inferioare
mean_upper_bound = np.mean(upper_bounds)  # Media limitelor superioare

print("Intervalul de predicție de 50% pentru valorile MEDV:", mean_lower_bound, mean_upper_bound)
