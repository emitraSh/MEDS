# what is maximum probable value for θ?
given we have result of a binomial distribution (binary variables) by plotting the conditional distribution across different thetas we can obtain most probable theta for the distribution.
- we have from theory maximum-likelihood value for theta is the mean of success. in here: 0.57
```doctest
import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt


def binom_theta(thetas, wins, N):

    thetas = np.asarray(thetas)
    multiplier = math.comb(N, wins)
    return multiplier * (thetas**wins) * ((1-thetas)**(N-wins))

thetas = np.linspace(0.0, 1.0, 101)


wins = 57
N = 100
y = binom_theta(thetas, wins , N)



res_df = pd.DataFrame({'theta':thetas, 'y':y})

plt.figure()
plt.plot(thetas, y)
plt.xlabel('theta')
plt.ylabel(f"P(Wins = {wins} | theta)")
plt.title("Binomial Likelihood as a Function of θ")
plt.grid(True)
plt.tight_layout()
plt.show()
```
given above computation:
# what is posterior P(θ|∑Y = 57) ?
given prior believe that each θ has the same probability(uniform), P(θ)= 1/11
- uniform prior can be removed from the posterior formula
```doctest
denominator = res_df['y'].sum()
res_df['posterior'] = (res_df['y']/denominator)

plt.figure()
plt.plot(res_df['posterior'], theta)
plt.xlabel('theta')
plt.xlabel(f"P(theta | wins)")
plt.title("Binomial Posterior as a Function of θ")
plt.grid(True)
plt.tight_layout()
plt.show()

```

- or theoretically we have that posterior of Binomial distribution converges to Beta distribution

```doctest
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

```


