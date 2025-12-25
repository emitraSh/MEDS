import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt


a_0 = 1
b_0 = 1

theta = np.linspace(0.001, 2, 1000)
# why 2? E[theta]= 1/mean = 0.5 (mean= 2)

prior_density = gamma.pdf(theta, a= a_0, scale=1/b_0)
posterior_N_25 = gamma.pdf(theta, a= 25+a_0, scale=1/(b_0+50))
posterior_N_100 = gamma.pdf(theta, a= 100+a_0, scale=1/(b_0+200))
posterior_N_1000 = gamma.pdf(theta, a= 1000+a_0, scale=1/(b_0+2000))

plt.figure(figsize = (8,5))
plt.plot(theta, prior_density,label = 'prior density', color = 'blue')
plt.plot(theta, posterior_N_25,label = 'posterior N=25', color = 'red')
plt.plot(theta, posterior_N_100,label = 'posterior N=100', color = 'green')
plt.plot(theta, posterior_N_1000,label = 'posterior N=1000', color = 'black')


plt.xlabel(r"$\theta$")
plt.ylabel("Density")
plt.title("Prior and Posterior Distributions for Different Sample Sizes")
plt.legend()
plt.grid(alpha=0.3)

plt.show()