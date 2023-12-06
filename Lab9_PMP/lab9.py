import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def run_model():
    # Încărcarea datelor
    data = pd.read_csv('Admission.csv')

    # Construirea modelului logistic folosind PyMC3
    with pm.Model() as model:
        # Parametrii modelului cu distribuții a priori slab informative
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        # Calculul probabilității folosind modelul logistic
        logits = beta0 + beta1 * data['GRE'] + beta2 * data['GPA']
        p = pm.math.sigmoid(logits)

        # Observațiile - admis/respins
        observed = pm.Bernoulli('observed', p=p, observed=data['Admission'])

        # Simularea eșantionului din distribuția a posteriori
        trace = pm.sample(1000, tune=1500)

    # 2. Determinarea Granței de Decizie
    # Calculul mediei coeficienților
    beta0_mean = np.mean(trace['beta0'])
    beta1_mean = np.mean(trace['beta1'])
    beta2_mean = np.mean(trace['beta2'])

    # Crearea unei grile de valori pentru GRE și GPA
    gre_vals = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
    gpa_vals = np.linspace(data['GPA'].min(), data['GPA'].max(), 100)
    gre_grid, gpa_grid = np.meshgrid(gre_vals, gpa_vals)

    # Calculul probabilităților de admitere pe grilă
    prob_grid = pm.math.sigmoid(beta0_mean + beta1_mean * gre_grid + beta2_mean * gpa_grid).eval()

    # Trasarea graniței de decizie și a intervalului HDI
    plt.contour(gre_grid, gpa_grid, prob_grid, levels=[0.5], colors='k') # Granița de decizie
    az.plot_hdi(gre_vals, prob_grid, hdi_prob=0.94, fill_kwargs={'alpha': 0.3}) # Intervalul HDI
    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.title('Granița de Decizie și Intervalul 94% HDI')
    plt.show()

    # 3. Interval HDI pentru un Student Specific (GRE: 550, GPA: 3.5)
    # Calculul probabilității și intervalul HDI pentru un student cu GRE: 550 și GPA: 3.5
    gre_val = 550
    gpa_val = 3.5
    student_prob = pm.math.sigmoid(beta0_mean + beta1_mean * gre_val + beta2_mean * gpa_val).eval()
    student_hdi = az.hdi(trace, var_names=['beta0', 'beta1', 'beta2'], hdi_prob=0.90)

    print("Probabilitatea de admitere pentru GRE 550 și GPA 3.5:", student_prob)
    print("Interval HDI 90% pentru GRE 550 și GPA 3.5:", student_hdi)

    # 4. Compararea a Doi Studenți (GRE: 500, GPA: 3.2)
    # Calculul probabilității și intervalul HDI pentru un student cu GRE: 500 și GPA: 3.2
    gre_val_2 = 500
    gpa_val_2 = 3.2
    student_prob_2 = pm.math.sigmoid(beta0_mean + beta1_mean * gre_val_2 + beta2_mean * gpa_val_2).eval()
    student_hdi_2 = az.hdi(trace, var_names=['beta0', 'beta1', 'beta2'], hdi_prob=0.90)

    print("Probabilitatea de admitere pentru GRE 500 și GPA 3.2:", student_prob_2)
    print("Interval HDI 90% pentru GRE 500 și GPA 3.2:", student_hdi_2)

    # Analiza diferențelor între cei doi studenți
    print("Diferența în probabilitatea de admitere între cei doi studenți este:", student_prob - student_prob_2)

if __name__ == '__main__':
    run_model()