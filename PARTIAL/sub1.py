from pgmpy.models import BayesianNetwork  # Actualizare conform avertizării
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definirea structurii rețelei Bayesiene
model = BayesianNetwork([('Start', 'FirstRound'), ('Start', 'SecondRound'), ('FirstRound', 'SecondRound')])

# CPD pentru 'Start'
cpd_start = TabularCPD(variable='Start', variable_card=2, values=[[0.5], [0.5]],
                       state_names={'Start': ['J0', 'J1']})

# CPD pentru 'FirstRound'
cpd_first_round = TabularCPD(variable='FirstRound', variable_card=2,
                             values=[[0.5, 0.5], [0.5, 0.5]],
                             evidence=['Start'], evidence_card=[2],
                             state_names={'FirstRound': ['0_steme', '1_stema'], 'Start': ['J0', 'J1']})

# CPD pentru 'SecondRound'
# Ajustarea formei matricei de valori pentru a se potrivi cu dimensiunile necesare (2 stări x 4 combinații posibile)
cpd_second_round = TabularCPD(variable='SecondRound', variable_card=2,
                              values=[[0.25, 0.5, 0.5, 0.75], [0.75, 0.5, 0.5, 0.25]],
                              evidence=['FirstRound', 'Start'], evidence_card=[2, 2],
                              state_names={'SecondRound': ['0_1_stema', '2_steme'],
                                           'FirstRound': ['0_steme', '1_stema'],
                                           'Start': ['J0', 'J1']})

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_start, cpd_first_round, cpd_second_round)

# Verificarea modelului
assert model.check_model()

# Crearea unui obiect de inferență
inference = VariableElimination(model)

# Calcularea probabilității pentru variabila 'Start', având observația că în 'SecondRound' s-a obținut o singură stemă
result = inference.query(variables=['Start'], evidence={'SecondRound': '0_1_stema'})

# Afișarea rezultatului
print(result)