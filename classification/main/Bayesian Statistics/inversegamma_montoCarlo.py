import matplotlib.pyplot as plt
from scipy.stats import gamma, invgamma
import numpy as np

sample_rv = gamma.rvs(a= 101, scale=1/201, size=1000)

sample_rv_transformed = 1/sample_rv

plt.figure(figsize=(8,5))
plt.hist(sample_rv_transformed, bins=100, density=True, label='Monte Carlo')

mu_grid = np.linspace(0.001, np.quantile(sample_rv_transformed,0.995), 1000)

inv_gamma_density = invgamma.pdf(mu_grid,a=101, scale=201)
plt.figure(figsize=(8,5))
plt.plot(mu_grid, inv_gamma_density, label='Inverse_Gamma', color='red')
plt.xlabel(r"$\mu$")
plt.ylabel("Density")
plt.title("Posterior of $\mu = 1/\\theta$")
plt.legend()
plt.grid(alpha=0.3)

plt.show()