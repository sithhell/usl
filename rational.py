
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def rational(x, ps, qs):
    nom = ps[0]
    denom = qs[0]
    degree = len(ps)
    for i in range(1, len(ps)):
        nom += ps[i] * (x ** i)
    for i in range(1, len(qs)):
        denom += qs[i] * (x ** i)
    return nom / denom

xs = np.linspace(0, 100, 1000)

ps = [.3, .1, .5, .1]
qs = [1.0, 1.0, 1.0, 1.0]
plt.plot(xs, [rational(x, ps, qs) for x in xs])
plt.show()
