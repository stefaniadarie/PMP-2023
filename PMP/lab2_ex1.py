# Doi mecanici schimbă filtrele de ulei pentru autoturisme într-un service. Timpul de servire este exponenţial
# cu parametrul λ1 = 4 hrs−1
# în cazul primului mecanic si λ2 = 6 hrs−1
# în cazul celui de al doilea. Deoarece al doilea
# mecanic este mai rapid, el serveşte de 1.5 ori mai mulţi clienţi decât partenerul său. Astfel când un client ajunge la rând,
# probabilitatea de a servit de primul mecanic este 40%. Fie X timpul de servire pentru un client.
# Generaţi 10000 de valori pentru X, şi în felul acesta estimaţi media şi deviaţia standard a lui X. Realizaţi un grafic al
# densităţii distribuţiei lui X.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Parametrii pentru distribuția exponențială
lambda1 = 4  # Primul mecanic
lambda2 = 6  # Al doilea mecanic

# Probabilitatea de a fi servit de primul mecanic
probabilitate_primul_mecanic = 0.4

# Numărul total de clienți
numar_clienti = 10000

# Generarea timpilor de servire pentru primul mecanic
timp_servire_primul_mecanic = expon(scale=1/lambda1).rvs(size=int(numar_clienti * probabilitate_primul_mecanic))

# Generarea timpilor de servire pentru al doilea mecanic
timp_servire_al_doilea_mecanic = expon(scale=1/lambda2).rvs(size=int(numar_clienti * (1 - probabilitate_primul_mecanic)))

# Combinarea timpilor de servire pentru ambii mecanici
timp_servire_total = np.concatenate((timp_servire_primul_mecanic, timp_servire_al_doilea_mecanic))

# Calculul mediei și deviației standard ale timpilor de servire
media_timp_servire = np.mean(timp_servire_total)
deviatia_standard_timp_servire = np.std(timp_servire_total)


print(f"Media timpului de servire: {media_timp_servire}")
print(f"Deviația standard a timpului de servire: {deviatia_standard_timp_servire}")

# Creez graficul densității distribuției timpului de servire
plt.hist(timp_servire_total, bins=50, density=True, alpha=0.6, color='g', label='Distribuția timpului de servire')
plt.xlabel('Timpul de servire')
plt.ylabel('Densitate')
plt.title('Densitatea distribuției timpului de servire')
plt.legend()
plt.show()
