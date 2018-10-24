
import math

from decimal import *
getcontext().prec = 30

def f(x, e, a, m, c, n, x_0):
    return e + a * (x**m) + c * ((x_0 - x)**n)

#print(f(Decimal(1), Decimal(3), Decimal(100), Decimal(0)))

import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(0, 1, 1000)


plt.plot(xs, [f(x, 6.33874337,  1.00000182,  1.84375361,  4.31011956,  6.52756008,  5.83339632) for x in xs])
plt.plot(xs, [f(x, 2.62653345,  2.07755379,  3.5735833,   5.92070409,  6.312991,    5.86790805) for x in xs])
plt.plot(xs, [f(x, 9.99999993e-01, 9.99999969e-01, 1.32846404e-08, 9.99999993e-01, 1.54532892e-06, 9.99998611e-01) for x in xs])
plt.show()
