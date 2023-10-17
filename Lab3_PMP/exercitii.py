from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Cutremur', 'Alarmă'), ('Incendiu', 'Alarmă')])

# Defining individual CPDs.
cpd_r = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_u = TabularCPD(variable='Incendiu', variable_card=2,
                          values=[[0.99, 0.03], [0.01, 0.97]],
                          evidence=['Cutremur'], evidence_card=[2])

# The CPD for C is defined using the conditional probabilities based on U and R
cpd_c = TabularCPD(variable='Alarmă', variable_card=2,
                        values=[[0.9999, 0.95, 0.02, 0.98],
                                [0.0001, 0.05, 0.98, 0.02]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_r, cpd_u, cpd_c)


# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
inferență = VariableElimination(model)

# P(Cutremur | Alarmă=1)
rezultat = inferență.query(variables=['Cutremur'], evidence={'Alarmă': 1})
print(rezultat)

#  P(Incendiu | Alarmă=0)
rezultat = inferență.query(variables=['Incendiu'], evidence={'Alarmă': 0})
print(rezultat)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()