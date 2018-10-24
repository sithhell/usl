
import numpy as np
from scipy.optimize import curve_fit
import math

class grain_model:
    def __init__(self, n, throughputs, grain_sizes):
        self.n = n
        throughputs = np.array(throughputs)
        self.t_mean = throughputs.mean()
        throughputs = throughputs / self.t_mean

        grain_sizes = np.array(grain_sizes)
        self.g_mean = grain_sizes.mean()
        grain_sizes = grain_sizes / self.g_mean

        if False:
            # This model is a rational polynomial
            self.model = lambda x, p0, p1, p2, q0, q1, q2: (p0 + x * (p1 + p2 * x)) / (q0 + x * (q1 + q2 * x))
            [[p0, p1, p2, q0, q1, q2], popt] = curve_fit(self.model, grain_sizes, throughputs, bounds=(0, np.inf))
            self.p0 = p0
            self.p1 = p1
            self.p2 = p2
            self.q0 = q0
            self.q1 = q1
            self.q2 = q2
        else:
            # This model is derived from the levy distribution
            self.model = lambda x, c, s, d: s * np.sqrt(c/(2*np.pi)) * np.exp(-c/(2. * x)) * x**(-1.5)

            [[c, s, d], popt] = curve_fit(self.model, grain_sizes, throughputs, bounds=(0, np.inf))
            self.c = c
            self.s = s
            self.d = d

    def __call__(self, grain_size):
        if False:
            return self.model(grain_size / self.g_mean, self.p0, self.p1, self.p2, self.q0, self.q1, self.q2) * self.t_mean
        else:
            return self.model(grain_size / self.g_mean, self.c, self.s, self.d) * self.t_mean
