import matplotlib
import numpy as np

matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)
import matplotlib.pylab as plt

plt.close("all")
plt.rcParams["figure.figsize"] = [8, 4]
from matplotlib.ticker import AutoMinorLocator

from gpyrn import kernels
from gpyrn import means as mean_functions
from gpyrn import variational

time = np.linspace(0, 100, 25)
y1 = 20 * np.sin(2 * np.pi * time / 31)
y1err = np.random.rand(25)

plt.figure()
plt.errorbar(time, y1, y1err, fmt="ob", markersize=7, label="y1")
plt.xlabel("Time (days)")
plt.ylabel("Measurements")
plt.grid(which="major", alpha=0.5)
plt.savefig("data.png", bbox_inches="tight")

############## 1 dataset - 1 node
gprn = variational.MeanFieldInference(1, time, y1, y1err)

nodes = [kernels.Periodic(15, 31, 0.5)]
weight = [kernels.SquaredExponential(1, 1)]
means = [mean_functions.Constant(0)]
jitter = [0.5]

elbo, m, v, _ = gprn.calculate_elbo(
    nodes, weight, means, jitter, max_iter=5000, mu="init", var="init"
)
print("ELBO =", elbo)

nodes = [kernels.Periodic(15, 31, 0.5)]
weight = [kernels.SquaredExponential(1, 100)]
means = [mean_functions.Constant(0)]
jitter = [0.5]

elbo, m, v, _ = gprn.calculate_elbo(
    nodes, weight, means, jitter, max_iter=5000, mu="init", var="init"
)
print("ELBO =", elbo)

tstar = np.linspace(time.min(), time.max(), 1000)
mean, _ = gprn.predict_from_variational_parameters(
    nodes, weight, means, jitter, tstar, m, v
)

plt.figure()
plt.errorbar(time, y1, y1err, fmt="ob", markersize=7, label="data")
plt.plot(tstar, mean, "--k", linewidth=2, label="predictive")
plt.xlabel("Time (days)")
plt.ylabel("Measurements")
plt.legend(loc="upper right", facecolor="white", framealpha=1, edgecolor="black")
plt.grid(which="major", alpha=0.5)
plt.savefig("dataAndPrediction.png", bbox_inches="tight")
