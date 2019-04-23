import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import scipy.optimize as opt


def gaussian(C):
    """
        return draw from a 0 mean gaussian with given convariance structure
        """
    return np.random.multivariate_normal(np.zeros(len(C)), C)


def sq_exp_kernel(x, x_d, theta=[1, 1]):
    """
        exponentiated quadratic kernel
        """
    sig, l = theta
    sqdist = (
        np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(x_d ** 2, 1) - 2 * np.dot(x, x_d.T)
    )

    return sig ** 2 * np.exp(-0.5 * (1 / l ** 2) * sqdist)


def logl(theta, x, y, u, x_test):

    x = x.reshape(-1, 1)
    K = sq_exp_kernel(x, x, theta) + np.eye(len(x)) * u ** 2
    y = y.reshape(-1, 1)

    L = (
        -0.5 * np.dot(np.dot(y.T, np.linalg.inv(K)), y)
        - 0.5 * np.log(np.linalg.det(K + 1e-6 * np.eye(len(x))))
        - 0.5 * len(x) * np.log(2 * np.pi)
    )[0][0]

    return L if np.isfinite(L) else 1e25


def predict_mean(x, y, u, x_s, theta):
    kss = sq_exp_kernel(x_s, x_s, theta)
    ksx = sq_exp_kernel(x_s, x, theta)
    kxs = sq_exp_kernel(x, x_s, theta)
    kxx = sq_exp_kernel(x, x, theta)
    o = u ** 2 * np.eye(len(x))

    L = np.linalg.inv(kxx + o)
    mu = ksx.dot(L).dot(y)
    V = np.diag(kss - ksx.dot(L).dot(ksx.T))

    return mu, V


def plot_all(x, y, u, x_s, theta):

    kss = sq_exp_kernel(x_s, x_s, theta)
    ksx = sq_exp_kernel(x_s, x, theta)
    kxs = sq_exp_kernel(x, x_s, theta)
    kxx = sq_exp_kernel(x, x, theta)
    inv = np.linalg.inv
    o = u ** 2 * np.eye(len(x))

    mu = ksx.dot(inv(kxx + o)).dot(y)
    C = kss - ksx.dot(inv(kxx + o)).dot(kxs)

    for i in range(10):
        plt.plot(
            x_s,
            np.random.multivariate_normal(mu.ravel(), C),
            color="#4682b4",
            lw=1,
            alpha=0.4,
        )

    plt.plot(x_s, mu, color="k")
    plt.errorbar(x.ravel(), y.ravel(), yerr=u, fmt="ok")
    plt.show()


def gp(x, y, u, x_s):

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_s = x_s.reshape(-1, 1)

    nll = lambda *args: -1 * logl(*args)
    result = opt.minimize(nll, [1, 1], args=(x, y, u, x_s))

    y_hat = predict_mean(x, y, u, x_s, result.x)
    return y_hat
