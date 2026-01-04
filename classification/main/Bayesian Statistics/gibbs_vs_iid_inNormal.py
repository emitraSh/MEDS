import numpy as np
from scipy.stats import norm, invgamma, probplot , gaussian_kde
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

sigma2_smpl_iid = invgamma.rvs(a= c_n , scale = np.sqrt(C_N), size = N_iid, random_state=rng)
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

def HPD_CI_interval(x, level= 0.95):
    x = np.sort(np.array(x))
    n= int(len(x))
    m= int(np.floor(level*n))   # number of data points that need to be included in CI
    width = x[m:] - x[:n-m] #this is amazing line of code, it selects intervals at both ends of x array
                            #that by subtracting each of them with respect to each other we get different widths(that include m data)
                            #and by detecting the shortest width we get the HPD ex: x[0:m],x[1:m+1], ... ,x[n-m,n]
    indx = np.argmin(width)
    return x[indx],x[indx+m]
lower_sigma2_iid , upper_sigma2_iid = HPD_CI_interval(sigma2_smpl_iid)
#lower_mu_iid , upper_mu_iid = HPD_CI_interval(mu_smpl_iid)
#lower_mu_gibbs , upper_mu_gibbs = HPD_CI_interval(mu_smpl_gibbs_ab)
lower_sigma2_gibbs , upper_sigma2_gibbs = HPD_CI_interval(sigma2_smpl_gibbs_ab)

# kde : kernel density, transforms discrete data into smooth shape
kde_sigma_iid = gaussian_kde(sigma2_smpl_iid)
#kde_mu_iid = gaussian_kde(mu_smpl_iid)
kde_sigma_gibbs = gaussian_kde(mu_smpl_gibbs_ab)
#kde_mu_gibbs = gaussian_kde(sigma2_smpl_gibbs_ab)

x_grid_sigma_iid = np.linspace(sigma2_smpl_iid.min(), sigma2_smpl_iid.max(), 1000)
#x_grid_mu_iid = np.linspace(mu_smpl_iid.min(), mu_smpl_iid.max(), 1000)
x_grid_sigma_gibbs = np.linspace(sigma2_smpl_gibbs_ab.min(), sigma2_smpl_gibbs_ab.max(), 1000)
#x_grid_mu_gibbs = np.linspace(mu_smpl_gibbs_ab.min(), mu_smpl_gibbs_ab.max(), 1000)

density_sigma_iid = kde_sigma_iid(x_grid_sigma_iid)
#density_mu_iid = kde_mu_iid(x_grid_mu_iid)
density_sigma_gibbs = kde_sigma_gibbs(x_grid_sigma_gibbs)
#density_mu_gibbs = kde_mu_gibbs(x_grid_mu_gibbs)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.plot(x_grid_sigma_iid, density_sigma_iid, color="black", linewidth=2)

mask = (x_grid_sigma_iid >= lower_sigma2_iid) & (x_grid_sigma_iid <= upper_sigma2_iid)
plt.fill_between(x_grid_sigma_iid[mask], density_sigma_iid[mask],
                 color="red", alpha=0.8)

# vertical lines at CI bounds
plt.axvline(lower_sigma2_iid, linestyle="--", color="black")
plt.axvline(upper_sigma2_iid, linestyle="--", color="black")

# labels
plt.xlabel(r"$\sigma_iid")
plt.ylabel("Density")

plt.title(f"{round(lower_sigma2_iid,4)} , {round(upper_sigma2_iid,4)} = 0.95$")

plt.subplot(1, 2, 2)
plt.plot(x_grid_sigma_gibbs, density_sigma_gibbs, color="black", linewidth=2)

mask = (x_grid_sigma_gibbs >= lower_sigma2_gibbs) & (x_grid_sigma_gibbs <= upper_sigma2_gibbs)
plt.fill_between(x_grid_sigma_gibbs[mask], density_sigma_gibbs[mask],
                 color="red", alpha=0.8)

# vertical lines at CI bounds
plt.axvline(lower_sigma2_gibbs, linestyle="--", color="black")
plt.axvline(upper_sigma2_gibbs, linestyle="--", color="black")

# labels
plt.xlabel(r"$\sigma_gibbs")
plt.ylabel("Density")

plt.title(f"{round(lower_sigma2_gibbs,4)},{round(upper_sigma2_gibbs,4)}")
plt.tight_layout()
plt.show()