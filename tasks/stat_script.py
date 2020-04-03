import numpy as np
import scipy.stats as st
from statsmodels.stats.diagnostic import lilliefors
import scikit_posthocs as sp
import pandas as pd

###############################################################################

mu = 5                                   # Média
sigma = 0.5                              # Dispersão
n = 30                                   # Número de instâncias
muu = 8
sigmaa = 0.2
sample = np.random.normal(mu, sigma, n)
samplee = np.random.normal(muu, sigmaa, n)
A = pd.DataFrame(sample)
B = pd.DataFrame(samplee)
teste = pd.concat([A, B], axis=1)
teste.columns = ['var1', 'var2']

###############################################################################
# Shapiro-Wilk

# Teste que permite verificar se uma amostra segue ou não uma distribuição nor-
# mal. p-valor < alpha (0.05) rejeita H0, rejeita a hipotese de que a amostra
# é normal.
###############################################################################
shapiro_stat, shapiro_p = st.shapiro(sample)

if shapiro_p > 0.05:
    print("Os dados são similares à uma distribuição normal !")
else:
    print("Os dados NÃO são similares à uma distribuição normal !")

###############################################################################
# Lilliefors

# É um teste de normalidade, corrigido em relação ao Kolmogorov-
# Smirnov para pequenos valores nas caudas das distribuições de probabilidade.
###############################################################################
stat, p = lilliefors(sample)
#print('statistics=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('Os dados são similares à uma distribuição normal !')
else:
    print('Os dados NÃO são similares à uma distribuição normal !')

###############################################################################
# Kruskal-Wallis

# O teste de Kruskal-Wallis é não paramétrico e pode ser utilizado para determinar
# se há diferenças estatisticamente significativas entre dois ou mais grupos de uma variável
# independente considerando contínuas ou ordinais.
###############################################################################
kruskal_stat, kruskal_p = st.kruskal(sample, samplee)
if kruskal_p > alpha:
    print('As medianas das populações são iguais.')
else:
    print('As medianas das populações NÃO são iguais.')
###############################################################################
# Dunn (Post-hoc)

# O teste de Dunn é um método estatístico Post Hoc não paramétrico usado para
# fazer um número específico de comparações entre grupos de dados e descobrir qual deles é
# significativo
###############################################################################
sp.posthoc_dunn(teste, p_adjust='holm', group_col='var1', val_col='var2')