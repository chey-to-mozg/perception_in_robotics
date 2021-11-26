"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import get_prediction
from tools.task import wrap_angle


class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization=False):
        super(PF, self).__init__(initial_state, alphas, beta)
        
        # TODO add here specific class variables for the PF
        self.M = num_particles
        ############
        if global_localization:
            self.X = []
            self.X.append(uniform(-100, 500, size=self.M))
            self.X.append(uniform(-100, 400, size=self.M))
            self.X.append(uniform(0, 2*np.pi, size=self.M))
            self.X = np.array(self.X).T
        else:
            self.X = []
            for i in range(3):
                self.X.append(gaussian.rvs(loc=self.mu[i], scale=np.sqrt(self.Sigma[i, i]), size=self.M))
            self.X = np.array(self.X).T
        ############\
        self.X_bar = []
        self.w = []


    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for m in range(self.M):
            self.X_bar.append(sample_from_odometry(self.X[m], u, self._alphas))

        self.X_bar = np.array(self.X_bar)


    def update(self, z):
        # TODO implement correction step
        bearing, lm_id = z[0], int(z[1])

        #closest observation - higher probability
        for m in range(self.M):
            h = get_observation(self.X[m], lm_id)[0]
            self.w.append(gaussian(0, np.sqrt(self._Q)).pdf(wrap_angle(h - bearing))) #

        self.w = np.array(self.w)
        self.w = self.w / sum(self.w)


        self.X = []

        # for m in range(self.M):
        #     u = uniform(0, 1)
        #     i = 0
        #     c = self.w[i]
        #     while(c < u):
        #         i += 1
        #         c += self.w[i]
        #     self.X.append(self.X_bar[i])

        r_v = uniform(0, 1 / self.M)

        c = self.w[0]
        i = 0

        #resampling
        for m in range(self.M):
            threshold = r_v + m / self.M
            while threshold > c:
                i += 1
                c += self.w[i]
            self.X.append(self.X_bar[i])

        self.X = np.array(self.X)

        self._state = get_gaussian_statistics(self.X)

        self.X_bar = []
        self.w = []
