## Gibbs Sampler for Categorical data (Dependency: Bivariate Normal)
- X is a categorical data that has n dimensions, n>2
- Y is a categorical data that has m dimensions, n>2
***
#### 1. How does the choice of pi,i ∈ {1,...,m × n} influence the speed of convergence?
- The aim is to converge to Stationary Markov distribution, meaning defining a unique distribution that can describe joint distribution based on conditional sampling.
- In the meantime, we can simply execute characteristics of marginal distributions based on samples.
- Gibbs sampler converges slowly when variables are strongly dependant since strong correlation does not allow for jumps in sampling and keeps samples in a loop.
___
Packages
```doctest
import numpy as np
```
step1 :     Constructing a Matrix with dependence among p_i's. For example "Bivariate Normal"

```doctest
Z = ((X-mu_x)**2/(2*sigma_x**2)
   + (Y-mu_y)**2/(2*sigma_y**2)
   - rho*(X-mu_x)*(Y-mu_y)/(sigma_x*sigma_y))
p = np.exp(-Z)
p/= p.sum()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, p, cmap='plasma',edgecolor='k' )
ax.set_title("Joint distribution")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

```
step 2 :     Construct Conditionals : X| Y=y and Y| X=x which are m × n shaped matrices.
```doctest
p_x_y = p / p.sum(axis=0, keepdims=True)
p_y_x = p / p.sum(axis=1, keepdims=True)

fig = plt.figure()
axy = fig.add_subplot(111, projection='3d')
axy.plot_surface(X, Y, p_x_y, cmap='plasma' ,edgecolor='k' )
axy.set_title("X | Y=y")
axy.set_xlabel("x")
axy.set_ylabel("y")
plt.show()

fig = plt.figure()
ayx = fig.add_subplot(111, projection='3d')
ayx.plot_surface(X, Y, p_y_x, cmap='plasma' ,edgecolor='k' )
ayx.set_title("Y | X=x")
ayx.set_xlabel("x")
ayx.set_ylabel("y")
plt.show()
```

step 3 : Apply Gibbs sampling on 1000000 samples with 100 first samples removed as burn in.

```doctest
N = 1000000
burn_in = 100

rng = np.random.default_rng(22)
rx = rng.random(N + burn_in)
ry = rng.random(N + burn_in)

x_sample = np.empty(N+burn_in, dtype=int)
y_sample = np.empty(N+burn_in, dtype=int)
xy_frequency = np.zeros((m,n),dtype=float)

x_sample[0] = 0
y_sample[0] = 0

for i in range(1, N + burn_in):

    #   Sample X | Y = y
    row = p[y_sample[i-1],:]
    row_norm = row / row.sum()
    x_sample[i] = np.searchsorted(np.cumsum(row_norm), rx[i])


    #   Sample Y | X = x
    column = p[:,x_sample[i-1]]
    column_norm = column / column.sum()
    y_sample[i] = np.searchsorted(np.cumsum(column_norm), ry[i])

    if i>burn_in:
        xy_frequency[y_sample[i],x_sample[i-1]] += 1
        xy_frequency[y_sample[i],x_sample[i]] += 1

xy_frequency /= xy_frequency.sum()


fig = plt.figure()
ac = fig.add_subplot(111, projection='3d')
ac.plot_surface(X, Y, xy_frequency, cmap='magma',edgecolor='k' )
ac.set_title("Empirical Stationary distribution (Gibbs Sampling)")
ac.set_xlabel("X")
ac.set_ylabel("Y ")
plt.show()
```
Step 4 : Construct conditional densities based on samples and compare the plot to the initial matrices

```doctest
a_x_y = xy_frequency / xy_frequency.sum(axis=0, keepdims=True)
a_y_x = xy_frequency / xy_frequency.sum(axis=1, keepdims=True)

fig = plt.figure()
asxy = fig.add_subplot(111, projection='3d')
asxy.plot_surface(X, Y, a_x_y, cmap='magma' ,edgecolor='k' )
asxy.set_title("X' | Y=y'")
asxy.set_xlabel("x'")
asxy.set_ylabel("y'")
plt.show()

fig = plt.figure()
asyx = fig.add_subplot(111, projection='3d')
asyx.plot_surface(X, Y, a_y_x, cmap='magma' ,edgecolor='k' )
asyx.set_title("Y' | X=x'")
asyx.set_xlabel("x'")
asyx.set_ylabel("y'")
plt.show()
```
