import numpy as np
import scipy.stats as stats
from statsmodels import stats

mu = 5
sigma = 0.5
n = 30
sample = np.random.normal(mu, sigma, n)


shapiro_stat, shapiro_p = stats.shapiro(sample)

if shapiro_p > 0.05:
    print("Os dados são similares à uma distribuição normal !")
else:
    print("Os dados NÃO são similares à uma distribuição normal !")


############
#  Shapiro-Wilk
#  Lilliefors 
#  Kruskal-Wallis
#  Dunn
############

aux = statsmodels.stats._lilliefors()
aux = stats.diagnostic.lilliefors(sample, dist='norm', pvalmethod='approx')