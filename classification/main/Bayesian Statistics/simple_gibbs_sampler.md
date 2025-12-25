# Gibbs sampler for  Bivariate random variable (A,B)'

- p(a,b)= c     if        a>0 , b>0 , 2a+b<1
- otherwise p(a,b) =0
- by computing ∬p(a,b) da db = 1 -> we have c= 4

using "a_1 = 0.1" and sampling we get an estimation of how the distribution should look like:
```doctest
import numpy as np
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


```
