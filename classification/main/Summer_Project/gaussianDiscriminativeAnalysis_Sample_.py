import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import multivariate_normal
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

@interact(mu1=(-3,3,0.1),  mu2=(-3,3.0,0.1), diagonal_1=(0,3.0,0.1), diagonal_2=(0,3.0,0.1), non_diagonal=(-3,3.0,0.1))
def visualize_multivariate_gaussian(mu1=0.0, mu2=0.0, diagonal_1=1, diagonal_2=1, non_diagonal=0):
    # This code snippet is taken from here [https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/]
    N = 300
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([mu1, mu2])
    Sigma = np.array([[ diagonal_1 , non_diagonal], [non_diagonal,  diagonal_2]])
    print("𝜇 = ", mu)
    print("Σ = ", Sigma)
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure(figsize=(15,10))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)

    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27, -21)

    plt.show()


def fit(x_train, y_train):
    m = y_train.shape[0]  # Number of training example
    # Reshapeing the training set
    x_train = x_train.reshape(m, -1)
    input_feature = x_train.shape[1]  # Number of input feature. In our case it's 4
    class_label = len(np.unique(y_train.reshape(-1)))  # Number of class. In our case its 3.

    # Start everything with zero first.
    # Mean for each class. Each row contains an individual class. And each of the class input is 4 dimenstional
    mu = np.zeros((class_label, input_feature))
    # Each row will conatain the covariance matrix of each class.
    # The covariance matrix is a square symettric matrix.
    # It indicates how each of the input feature varies with each other.
    sigma = np.zeros((class_label, input_feature, input_feature))
    # Prior probability of each class.
    # Its the measure of knowing the likelihood of any class before seeing the input data.
    phi = np.zeros(class_label)

    for label in range(class_label):
        # Seperate all the training data for a single class
        indices = (y_train == label)

        phi[label] = float(np.sum(indices)) / m
        mu[label] = np.mean(x_train[indices, :], axis=0)
        # Instead of writting the equation we used numpy covariance function.
        sigma[label] = np.cov(x_train[indices, :], rowvar=0)

    return phi, mu, sigma