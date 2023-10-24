import numpy as np
from matplotlib.patheffects import Normal
from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
from sympy.stats import Poisson, Exponential

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('Trafic', 'Plata'), ('Plata', 'Gatit')])

# Defining individual CPDs.
cpd_t = TabularCPD(variable='Trafic', variable_card=2, values=[[0.0], [1.0]])  # Poisson distribution
cpd_p = TabularCPD(variable=['Plata'], variable_card=1, values=lambda x: 1/(0.5*np.sqrt(2*np.pi))*np.exp(-0.5*((x-2)/0.5)**2)) # Normal distribution
cpd_g = TabularCPD(variable='Gatit', variable_card=1)


# Associating the CPDs with the network
model.add_cpds(cpd_t, cpd_p, cpd_g)


# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
inference = VariableElimination(model)


rezultat = inference.query(variables=['Plata'], evidence={'Trafic': 25})
print(rezultat)

timp_mediu_plata = 2  # Timpul mediu de plasare și plată
timp_mediu_servire = 1 / max # Timpul mediu de servire a comenzii de gătit
timp_mediu_asteptare = timp_mediu_plata + timp_mediu_servire
print(f"Timpul mediu de așteptare pentru servirea unui client: {timp_mediu_asteptare}")

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()