
import numpy as np
from scipy.optimize import curve_fit
import math

class usl:

    def __init__(self, N, X):
        self.alpha = 0.0
        self.beta = 0.0
        self.y = 0.0
        self.capacity = lambda N, alpha, beta, y: (y * N) / (1 + alpha * (N-1) + beta * N * (N-1))

        popt, pcov = curve_fit(self.capacity, N, X, bounds=(0,np.inf))

        self.alpha = popt[0]
        self.beta = popt[1]
        self.y = popt[2]

        #if self.beta == 0.0 or self.alpha < 0:
        #    self.N_max = X[-1]
        #else:
        #    self.N_max = int(math.sqrt((1-self.alpha)/self.beta))
#
#        if self.alpha == 0.0:
 #           self.N_opt = X[-1]
 #       else:
 #           self.N_opt = int(math.ceil(1/self.alpha))

        #p = np.poly1d([a, b, 0])
        #yhat = p(X)
        #ybar = np.sum(y)/len(y)
        #ssreg = np.sum((yhat - ybar)**2)
        #sstot = np.sum((y - ybar)**2)
        #self.R_sq = ssreg/sstot;

    def summary(self):
  #      print('N_opt = %s, N_max = %s' % (self.N_max, self.N_opt))
        print('alpha = %s, beta = %s, lambda = %s' % (self.alpha, self.beta, self.y))
        print('N / (1 + %s(N-1) + %sN(N-1))' % (self.alpha, self.beta))

    def throughput(self, N, mode = None):

        if mode is None:
            return self.capacity(N, self.alpha, self.beta, self.y)
        if mode is 'amdahl':
            return self.capacity(N, self.alpha, 0.0, self.y)
        if mode == 'gustafson':
            return self.y * ((1 - self.alpha) * N + self.alpha)
        if mode == 'ideal':
            return self.capacity(N, 0.0, 0.0, self.y)
        raise 'mode has to be either unset, \'amdahl\', \'gustafson\' or \'ideal\''

    def latency(self, N, mode = None):
        if mode is None:
            result = float(N / self.throughput(N = N, mode=None))
            return result
        if mode is 'amdahl':
            result = float(N / self.throughput(N = N, mode='amdahl'))
            return result
        if mode is 'ideal':
            result = float(N / self.throughput(N = N, mode='ideal'))
            return result
        raise 'mode has to be either unset, \'amdahl\', \'ideal\''

