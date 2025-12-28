## Gibbs Sampler for Categorical data
- X is a categorical data that has n dimensions, n>2
- Y is a categorical data that has m dimensions, n>2
***
#### 1. How does the choice of pi,i ∈ {1,...,m × n} influence the speed of convergence?
- The aim is to converge to Stationary Markov distribution, meaning defining a unique distribution that can describe joint distribution based on conditional sampling.
- In the meantime, we can simply execute characteristics of marginal distributions based on samples.
- Gibbs sampler converges slowly when variables are strongly dependant since strong correlation does not allow for jumps in sampling and keeps samples in a loop.
#### 2.  Show an example that breaks Gibbs sampler.
___
Packages
```doctest
import numpy as np
```
step1 :     Constructing a Matrix with dependence among p_i's.
```doctest
n =

```