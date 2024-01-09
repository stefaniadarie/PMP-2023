import numpy as np
import scipy.stats as stats

def metropolis(data, prior_params, iterations=10000):
    alpha, beta = prior_params['alpha'], prior_params['beta']
    current = stats.beta.rvs(alpha, beta)  # Punct de start aleatoriu
    samples = []

    for i in range(iterations):
        proposal = current + np.random.normal(0, 0.1)  # Propunere nouă
        proposal = min(1, max(proposal, 0))  # Asigurăm că propunerea este între 0 și 1

        # Calculez probabilitățile pentru starea curentă și propunere
        likelihood_current = stats.binom.pmf(data[0], data[0] + data[1], current)
        likelihood_proposal = stats.binom.pmf(data[0], data[0] + data[1], proposal)

        prior_current = stats.beta.pdf(current, alpha, beta)
        prior_proposal = stats.beta.pdf(proposal, alpha, beta)

        p_accept = (likelihood_proposal * prior_proposal) / (likelihood_current * prior_current)

        if np.random.rand() < p_accept:
            current = proposal

        samples.append(current)

    return np.array(samples)

# Date și parametrii a priori pentru modelul beta-binomial
data = [successes, failures]
prior_params = {'alpha': 1, 'beta': 1}

# Rulați Metropolis
samples = metropolis(data, prior_params)