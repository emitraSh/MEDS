import numpy as np
from scipy.stats import norm, invgamma, probplot
import matplotlib.pyplot as plt
from statsmodels.api import qqplot



rng = np.random.default_rng()

# __ Generating Normally distributed data to apply the models on them __

mu = 0
sigma2 = 2.5048
N = 50
y = rng.normal(mu, np.sqrt(sigma2), N)

# __ Data Summaries

y_mean = np.mean(y)
sample_var = np.var(y, ddof=1)
N_iid = 1000

# _____                         generation **   iid    ** pairs                      _____

c_n = (N -1)/2
C_N = c_n * sample_var

sigma2_smpl_iid = invgamma.rvs(a= c_n , scale = C_N, size = N_iid, random_state=rng)
mu_smpl_iid = norm.rvs(loc = y_mean, scale = np.sqrt(sigma2_smpl_iid/N), size = N_iid, random_state=rng)


# _____                         generation **   gibbs    ** pairs                    _____

N_gibbs = 1101
burn_in = 100
mu_smpl_gibbs = np.empty(N_gibbs)
sigma2_smpl_gibbs = np.empty(N_gibbs)

mu_smpl_gibbs[0]= 1
sigma2_smpl_gibbs[0]= 1

for j in range(1,N_gibbs):
    mu_smpl_gibbs[j] = norm.rvs(loc = y_mean,
                                scale = np.sqrt(sigma2_smpl_gibbs[j-1]/N),
                                random_state=rng)

    sigma2_smpl_gibbs[j] = invgamma.rvs(a= c_n ,
                                        scale = np.sqrt((np.sum((y - mu_smpl_gibbs[j])**2))/2),
                                        random_state=rng)

mu_smpl_gibbs_ab=mu_smpl_gibbs[burn_in:]
sigma2_smpl_gibbs_ab=sigma2_smpl_gibbs[burn_in:]


new_y_iid = norm.rvs(loc = mu_smpl_iid,
                     scale = np.sqrt(sigma2_smpl_iid/N),
                     random_state=rng)

new_y_gibbs = norm.rvs(loc = mu_smpl_gibbs_ab,
                     scale = np.sqrt(sigma2_smpl_gibbs_ab/N),
                     random_state=rng)



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
qqplot(new_y_iid, line='45', fit=True, ax=plt.gca())
plt.title("Q-Q Plot (iid draws)")

plt.subplot(1, 2, 2)
qqplot(new_y_gibbs, line='45', fit=True, ax=plt.gca())
plt.title("Q-Q Plot (gibbs draws)")

plt.tight_layout()
plt.show()

# Q-Q Plot using scipy
"""plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
probplot(new_y_iid, dist="norm", plot=plt)
plt.title("SciPy ProbPlot (iid draws)")

plt.subplot(1, 2, 2)
probplot(new_y_gibbs, dist="norm", plot=plt)
plt.title("SciPy ProbPlot (gibbs draws)")

plt.tight_layout()
plt.show()
"""
# Trace plot for mu gibbs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu_smpl_gibbs, linewidth=0.7)
plt.xlabel("Iteration")
plt.ylabel(r"$\mu$")
plt.title("Trace plot for $\mu$ Gibbs")


# Trace plot for sigma^2 gibbs
plt.subplot(1, 2, 2)
plt.plot(sigma2_smpl_gibbs, linewidth=0.7)
plt.xlabel("Iteration")
plt.ylabel(r"$\sigma^2$ ")
plt.title("Trace plot for $\sigma^2$ Gibbs")
plt.tight_layout()
plt.show()


# Trace plot for mu iid
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mu_smpl_iid, linewidth=0.7)
plt.xlabel("Iteration")
plt.ylabel(r"$\mu$")
plt.title("Trace plot for $\mu$ iid")


# Trace plot for sigma^2 iid
plt.subplot(1, 2, 2)
plt.plot(sigma2_smpl_iid, linewidth=0.7)
plt.xlabel("Iteration")
plt.ylabel(r"$\sigma^2$ ")
plt.title("Trace plot for $\sigma^2$ iid")
plt.tight_layout()
plt.show()


