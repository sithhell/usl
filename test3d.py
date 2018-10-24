import numpy as np
import usl
import grain_model

#N = [1, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
#X = [20, 78, 130, 170, 190, 200, 210, 230, 260, 280, 310]
#N = [ 1, 18, 36, 72, 108, 144, 216 ]
#X = [ 65, 996, 1652, 1853, 1829, 1775, 1702 ]
N = [1, 2, 4, 8, 16]
TASKS = 120000.0

grain_sizes = []
Ts = []
Xs = []

#grain_sizes.append(1000*1e-9)
#Ts.append([729511484., 1511331027., 3410577716., 9798453072., 30333467571.])
#Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

#grain_sizes.append(5000*1e-9)
#Ts.append([741342331., 379401105., 217216942., 143431892., 123450747.])
#Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

grain_sizes.append(10000*1e-9)
Ts.append([1328110496., 688804661., 367476057., 223567014., 165615113.])
Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

#grain_sizes.append(50000*1e-9)
#Ts.append([6127683814., 3085399794., 1572583359., 823277974., 464040631.])
#Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

#grain_sizes.append(100000*1e-9)
#Ts.append([12123977808., 6087994710., 3073404944., 1578404027., 836524955.])
#Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])

#grain_sizes.append(500000*1e-9)
#Ts.append([60152155532., 30102326746., 15074890648., 7579713056., 3839874772.])
#Xs.append([ TASKS / (t * 1e-9) for t in Ts[-1] ])


models = [ usl.usl(N, X) for X in Xs ]
grain_models = {}

#plt.subplot(323)
for idx in range(len(N)):
    #plt.figure(2 + idx)
    #plt.title('Varying Grain Sizes, N=%s' % N[idx])

    grain_models[N[idx]] = grain_model.grain_model(N[idx], [ x[idx] for x in Xs ], grain_sizes)

nn = np.linspace(N[0], N[-1], 1000)

grain_models_fine = {}
for n in nn:
    X = [model.throughput(n) for model in models]
    grain_models_fine[n] = grain_model.grain_model(n, X, grain_sizes)

import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.title('Throughput')
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


rs = np.linspace(0, 0.0001, 1000)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure(2)
ax = fig.gca(projection='3d')

#for idx in range(len(N)):

print('----')

X_scale = 1000000
G_scale = 1e6

for i in range(len(grain_sizes)):
    ax.scatter(N, np.full(len(N), grain_sizes[i]*G_scale), [Xs[i][idx]/X_scale for idx in range(len(N))], marker='o', label='Original Data, grainsize=%s' % grain_sizes[i])
    #ax.plot(nn, np.full(len(nn), grain_sizes[i]), models[i].throughput(nn), label='USL Fitted, grainsize=%s' % grain_sizes[i])

#for i in range(len(N)):
#    ax.plot(np.full(len(rs[1:]), N[i]), rs[1:], grain_models[i](rs[1:]), label='Levy Fitted, N=%s' % N[i])

#for grain_model in grain_models_fine:
    #ax.plot(np.full(len(rs[1:]), grain_model.n), rs[1:], grain_model(rs[1:]))#, label='Levy Fitted, N=%s' % grain_model.n)

def surface_plot(N, G):
    result = []
    for (nn, gg) in zip(N, G):
        z = []
        for (n, g) in zip(nn, gg):
            z.append(grain_models_fine[n](g)/X_scale)
        result.append(z)
    return result


XX, YY = np.meshgrid(nn, rs[1:])
Z = surface_plot(XX, YY)

ax.legend()

surf = ax.plot_surface(XX, YY*G_scale, Z, cmap=cm.coolwarm, linewidth=0, label='Grain Size Predictions', rcount=50, ccount=150)

ax.set_xlabel('Number of Cores')
ax.set_ylabel(r'Grain Size [$\mu s$]')
ax.set_zlabel('Tasks/s')
ax.set_zlim(0)
ax.zaxis.set_major_locator(LinearLocator(15))
ax.zaxis.set_major_formatter(FormatStrFormatter(r'$ 10^6 \cdot %.02f$'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
