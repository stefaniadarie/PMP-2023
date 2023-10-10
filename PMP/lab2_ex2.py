# Patru servere web oferă acelaşi serviciu (web) clienţilor . Timpul necesar procesării unei cereri (request)
# HTTP este distribuit Γ(4, 3) pe primul server, Γ(4, 2) pe cel de-al doilea, Γ(5, 2) pe cel de-al treilea, şi Γ(5, 3) pe cel de-al
# patrulea (în milisecunde). La această durată se adaugă latenţa dintre client şi serverele pe Internet, care are o distribuţie
# exponenţială cu λ = 4 (în miliseconde−1). Se ştie că un client este direcţionat către primul server cu probabilitatea 0.25,
# către al doilea cu probabilitatea 0.25, iar către al treilea server cu probabilitatea 0.30. Estimaţi probabilitatea ca timpul
# necesar servirii unui client, notat cu X, (de la lansarea cererii până la primirea răspunsului) să fie mai mare decât 3
# milisecunde. Realizaţi un grafic al densităţii distribuţiei lui X.
# Notă: Distribuţia Γ(α, λ) se poate apela cu stats.gamma(α,0,1/λ) sau stats.gamma(α,scale=1/λ).

import numpy as np
from scipy.stats import gamma, expon

alpha = [4, 4, 5, 5]
lambda_ = [3, 2, 2, 3]

# direcționare către fiecare server
probabilitati_servere = [0.25, 0.25, 0.30, 0.20]

lambda_latenta = 4

# calculul probabilității P(X > 3) pentru fiecare server
def probabilitate_X_mai_mare_de_3(alpha, lambda_, lambda_latenta):
    timp_procesare = gamma(alpha, scale=1/lambda_).rvs()
    latența = expon(scale=1/lambda_latenta).rvs()
    X = timp_procesare + latența
    return X > 3

# calculul probabilității totale
probabilitate_totala = sum(probabilitate_X_mai_mare_de_3(alpha[i], lambda_[i], lambda_latenta) * probabilitati_servere[i] for i in range(4))

print(f"Probabilitatea ca timpul de servire (X) să fie mai mare de 3 ms este: {probabilitate_totala}")

# Trasarea graficului
#