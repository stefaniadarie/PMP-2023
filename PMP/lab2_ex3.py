# Se consideră un experiment aleator prin aruncarea de 10 ori a două monezi, una nemăsluită, cealaltă cu
# probabilitatea de 0.3 de a obţine stemă. Să se genereze 100 de rezultate independente ale acestui experiment şi astfel
# să se determine grafic distribuţiile variabilelor aleatoare care numără rezultatele posibile în cele 10 aruncări (câte una
# pentru fiecare rezultat posibil: ss, sb, bs, bb).


import numpy as np
import matplotlib.pyplot as plt

# definesc probabilitatea de a obține stemă pentru a doua monedă
probabilitate_stema = 0.3


# aruncarea unei monede
def aruncare_moneda(probabilitate_stema):
    rezultat = np.random.choice(['s', 'b'], p=[1 - probabilitate_stema, probabilitate_stema])
    return rezultat


# 10 aruncări și numărarea rezultatelor posibile
def experiment_10_aruncari(probabilitate_stema):
    rezultate = ""
    for _ in range(10):
        rezultate += aruncare_moneda(probabilitate_stema)
    return rezultate

# generez 100 de rezultate independente ale experimentului
rezultate_experiment = [experiment_10_aruncari(probabilitate_stema) for _ in range(100)]

# rezultate posibile în cele 10 aruncări
numar_ss = rezultate_experiment.count("ss")
numar_sb = rezultate_experiment.count("sb")
numar_bs = rezultate_experiment.count("bs")
numar_bb = rezultate_experiment.count("bb")

# construiesc graficul //sau incerc :))
rezultate_posibile = ["ss", "sb", "bs", "bb"]
numere = [numar_ss, numar_sb, numar_bs, numar_bb]

plt.bar(rezultate_posibile, numere)
plt.xlabel("Rezultate Posibile")
plt.ylabel("Număr de apariții")
plt.title("Distribuția rezultatelor în cele 10 aruncări")
plt.show()
