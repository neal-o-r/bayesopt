from GP import gp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

np.random.seed(123)
u = 0.5


def f(x):
    return 5 * np.sin(3 * x) + 0.3 * x ** 2 - 2 * x


def get_data(x=None):
    if x is None:
        x = np.random.uniform(-2, 2)

    y = f(x) + np.random.randn() * u
    return x, y


def acquisition(y, x_s, y_s, sig):
    plus = lambda x: np.maximum(x, 0)

    delta = y_s.ravel() - np.max(y)
    pi = (
        plus(delta)
        + sig * norm.pdf(delta / sig)
        - np.abs(delta) * norm.cdf(delta / sig)
    )

    return pi


def plot_all(x, y, x_s, y_s, sig, p, i):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.errorbar(x, y, yerr=u, fmt="ok")
    ax1.plot(x_s, y_s, "#4682b4")
    ax1.plot(x_s, f(x_s), "k", alpha=0.7)

    ax1.fill_between(
        x_s, y_s.ravel() - sig, y_s.ravel() + sig, color="#dddddd", alpha=0.5
    )

    ax1.vlines(-1.6375, -100, 100, linestyles="dashed", lw=1)
    ax1.set_ylim(-10, 10)
    ax1.set_xlim(-2, 2)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])

    ax2.plot(x_s, p, "r")
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig(f"gif/{str(i).zfill(2)}.png")


if __name__ == "__main__":
    x_s = np.linspace(-2, 2, 100)

    x_all, y_all = [], []
    p = np.random.rand(len(x_s))
    i = 0
    while i < 25:

        i += 1
        x, y = get_data(x_s[np.argmax(p)])
        x_all.append(x)
        y_all.append(y)

        y_s, sig = gp(np.array(x_all), np.array(y_all), u, x_s)
        p = acquisition(y_all, x_s, y_s, sig)

        plot_all(x_all, y_all, x_s, y_s, sig, p, i)
