import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n = 15
m = 25
px = np.arange(n)
py = np.arange(m)
X, Y = np.meshgrid(px, py)

mu_x = n/2
mu_y = m/2
sigma_x = n/6
sigma_y = m/6
rho = 0.7 # the dependencies among x and y

#_________  Using Bivariate Normal to impose dependencies among categories  _______________
Z = ((X-mu_x)**2/(2*sigma_x**2)
   + (Y-mu_y)**2/(2*sigma_y**2)
   - rho*(X-mu_x)*(Y-mu_y)/(sigma_x*sigma_y))
p = np.exp(-Z)
p/= p.sum()

print(f'_____________ {p.shape}________________')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, p, cmap='plasma',edgecolor='k' )
ax.set_title("Joint distribution")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

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

# ___________ Gibbs _________

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

    if x_sample[i] >= n:
        print("⚠️ BAD INDEX x[i] =", x_sample[i], " at i =", i , row_norm)
        break

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



