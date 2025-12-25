# There is an experiment going on.
##                       We know :
# random variable follows Binomial Distribution, Number of trials is Poisson(mu), probability is Beta(a,b), number of success y=5
##                 We don't know :
# number of trials N , parameter p defining success
#            method              :            Grid Evaluation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson,binom,beta

# Hyperparameters:

y= 5
a,b = 2,3 # we dont have it
mu = 20 # we dont have it

###

n_grid = np.arange(y+1, 50)
p_grid = np.linspace(0.001,0.99,100)
posterior_mtrx = np.zeros((len(n_grid),len(p_grid)))

n_prior= poisson.pmf(n_grid, mu)
p_prior= beta.pdf(p_grid, a, b = b)

for i, N in enumerate(n_grid):
    for j, p in enumerate(p_grid):
        likelihood = binom.pmf(y , N, p)
        posterior_mtrx[i,j] = likelihood * n_prior[i] * p_prior[j]

# normalizing :
posterior_mtrx = posterior_mtrx / posterior_mtrx.sum()

# marginal posteriors
p_N_given_py = posterior_mtrx.sum(axis = 1) #sum over p
p_p_given_Ny = posterior_mtrx.sum(axis = 0) #sum over N


plt.figure(figsize=(8,5))
plt.imshow(posterior_mtrx, aspect= "auto", origin='lower',
           extent=[p_grid.min(), p_grid.max(), n_grid.min(), n_grid.max()]) #heatmap?
plt.colorbar(label= 'Posterior Density')
plt.xlabel("p~B(a,b)")
plt.ylabel("N~Poisson(mu)")
plt.title("P(y=5 | N, p)")
plt.show()


plt.figure(figsize=(6,4))
plt.bar(n_grid, p_N_given_py)
plt.title("Marginal Posterior Density  of N")
plt.xlabel("N")
plt.ylabel("Probability")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(p_grid, p_p_given_Ny)
plt.title("Marginal Posterior Density  of p")
plt.xlabel("p")
plt.ylabel("Probability")
plt.show()









