"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(16)

def gibbs_sampler(n_iter=1000, burn_in=300, a0=0.1):

    a = np.zeros(n_iter+1)
    b = np.zeros(n_iter+1)

    a[0] = a0
    b[0] = np.random.uniform(0,1- 2*a0)

    for t in range(1, n_iter):
        a[t] = np.random.uniform(0,(1- b[t-1])/2.0)
        b[t] = np.random.uniform(0,1- (2.0*a[t-1]))

    return a[burn_in + 1:], b[burn_in + 1:]

a_rv, b_rv = gibbs_sampler(n_iter=1000, burn_in=111, a0=0.1)
res_df = pd.DataFrame({'a':a_rv, 'b':b_rv})

c= np.ones_like(a_rv)*4 # arrey of 1s with similar shape as a multiplied by 4 {4,4,4,4...}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(a_rv, b_rv, c, s=5)
ax.set_xlabel("A")
ax.set_ylabel("B")
ax.set_zlabel("C")

plt.show()
"""

#question 5
import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

"""
def binom_theta(thetas, wins, N):

    thetas = np.asarray(thetas)
    multiplier = math.comb(N, wins)
    return multiplier * (thetas**wins) * ((1-thetas)**(N-wins))

thetas = np.linspace(0.0, 1.0, 11)


wins = 57
N = 100
y = binom_theta(thetas, wins , N)



res_df = pd.DataFrame({'theta':thetas, 'y':y})"""

"""plt.figure()
plt.plot(thetas, y)
plt.xlabel('theta')
plt.ylabel(f"P(Wins = {wins} | theta)")
plt.title("Binomial Likelihood as a Function of θ")
plt.grid(True)
plt.tight_layout()
plt.show()
"""

"""denominator = res_df['y'].sum()
res_df['posterior'] = (res_df['y']/denominator)

plt.figure()
plt.plot(thetas,res_df['posterior'])
plt.xlabel('theta')
plt.ylabel(f"P(theta | wins)")
plt.title("Binomial Posterior as a Function of θ")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
thetas = np.linspace(0.0,1.0,100)
pdf =beta.pdf(thetas,58,44)


plt.figure()
plt.plot(thetas, pdf)
plt.xlabel("theta")
plt.ylabel("density")
plt.title(f"Beta({58}, {44}) distribution")
plt.tight_layout()
plt.show()
