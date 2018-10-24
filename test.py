import numpy as np
#import math

import usl

#N = [1, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
#X = [20, 78, 130, 170, 190, 200, 210, 230, 260, 280, 310]
#N = [ 1, 18, 36, 72, 108, 144, 216 ]
#X = [ 65, 996, 1652, 1853, 1829, 1775, 1702 ]
N = [1, 2, 4, 8, 16]
TASKS = 120000.0

grain_sizes = []
Ts = []
Xs = []

grain_sizes.append(1000*1e-9)
Ts.append([729511484., 1511331027., 3410577716., 9798453072., 30333467571.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(5000*1e-9)
Ts.append([741342331., 379401105., 217216942., 143431892., 123450747.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(10000*1e-9)
Ts.append([1328110496., 688804661., 367476057., 223567014., 165615113.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(50000*1e-9)
Ts.append([6127683814., 3085399794., 1572583359., 823277974., 464040631.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(100000*1e-9)
Ts.append([12123977808., 6087994710., 3073404944., 1578404027., 836524955.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(500000*1e-9)
Ts.append([60152155532., 30102326746., 15074890648., 7579713056., 3839874772.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])


models = [ usl.usl(N, X) for X in Xs ]

import sys

nn = np.linspace(N[0], N[-1] * 8)

import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.title('Throughput')
plt.plot(N, X, 'o', label='Original data', markersize=10)
for idx in range(len(models)):
    plt.scatter(N, Xs[idx], marker='o', label='Original data grainsize=%s' % grain_sizes[idx])
    plt.plot(nn, [models[idx].throughput(n) for n in nn], label='Fitted line grainsize=%s' % grain_sizes[idx])

    #plt.plot(nn, [model.throughput(n, mode='amdahl') for n in nn], 'g--', label='Fitted line (Amdahl)')
    #plt.plot(N, [model.throughput(n, mode='gustafson') for n in N], 'b', label='Fitted line (Gustafson)')
    #plt.plot(N, [model.throughput(n, mode='ideal') for n in N], 'k--', label='Ideal Scaling')

plt.legend(loc='center right', bbox_to_anchor=(1,0.5), fancybox = True)

plt.subplot(212)
plt.title('Average Latency = Overhead + Grain Size')
for idx in range(len(models)):
    plt.plot(nn, [models[idx].latency(n) for n in nn], label='Latency, grainsize=%s' % grain_sizes[idx])
plt.legend()


from scipy.optimize import curve_fit
import numpy as np
import math

def weibull(xs, beta, n, x_0):
    result = []
    for x in xs:
        #result.append(a * x**2 + b * x + c)
        if n - x_0 == 0.0:
            return np.inf
        tmp = (x - x_0)/(n - x_0)
        tmp_result = beta / (n - x_0)
        tmp_result *= tmp**(beta - 1)
        tmp_result *= math.exp(-tmp**beta)
        result.append(tmp_result)
    return result

def bathtub(x, e, a, m, c, n, x_0):
    return e + a * (x**m) + c * ((x_0 - x)**n)

def rational(x, p0, p1, p2, q0, q1, q2):
    return (p0 + x * (p1 + p2 * x)) / (q0 + x * (q1 + q2 * x))

def levy(x, c, s, d):
    return s * np.sqrt(c/(2*np.pi)) * np.exp(-c/(2. * x)) * x**(-1.5)

def patrick1(x, a, b):
    return a * b * np.exp(-a * x**2) * x

def patrick2(x, a, b):
    return b * np.exp(-a * x**2) * x

def patrick3(x, a, b):
    return a * b * np.exp(-a/b * x**2) * x

def patrick4(x, a, b):
    return a * b * np.exp(-a/(b * x**2)) * x

import scipy.interpolate as interp
import scipy.optimize as optimize
rs = np.linspace(0, grain_sizes[-1], 10000)

grain_model = []

#plt.subplot(323)
for idx in range(len(N)):
    plt.figure(2 + idx)
    plt.title('Varying Grain Sizes, N=%s' % N[idx])
    X = [ x[idx] for x in Xs ]
    X = np.array(X)
    X_mean = X.mean()
    X = X / X_mean
    G = np.array(grain_sizes)
    G_mean = G.mean()
    G = G / G_mean
    plt.plot(grain_sizes, X * X_mean, 'ok', label='Original Data, N = %s' % N[idx])
    plt.plot(grain_sizes, X * X_mean, 'k--')

    [popt, pcov] = curve_fit(rational, G, X, bounds=(0, np.inf))
    plt.plot(rs, rational(rs / G_mean, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) * X_mean, label='fitted, n = %s, P(x)/Q(x)' % (N[idx]))

    [popt, pcov] = curve_fit(levy, G, X, bounds=(0, np.inf))
    plt.plot(rs[1:], levy(rs[1:] / G_mean, popt[0], popt[1], popt[2]) * X_mean, label='Fitted, n = %s, levy distribution' % (N[idx]))

    plt.legend()

    grain_model.append

plt.show()
