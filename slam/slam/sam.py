"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from scipy.stats import norm as gaussian
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import *
from matplotlib import pyplot as plt
from tools.plot import plot2dcov

print_in_console = False  # print estimated states

batch_solution = False  # For 2.E


class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas

        self.state = initial_state

        self.observed_lms = {}
        
        self.nodes = []

        self.errors = []

        # 1.A
        if print_in_console:
            print('#####init#######')
        self.graph = mrob.FGraph()

        x_0 = []
        for i in range(3):
            x_0.append(gaussian.rvs(loc=self.mu[i], scale=np.sqrt(self.Sigma[i, i])))

        if print_in_console:
            print(f'initial state: {x_0}')

        n = self.graph.add_node_pose_2d(x_0)
        if print_in_console:
            print(f'node id: {n}')

        W = inv(self.Sigma)

        self.graph.add_factor_1pose_2d(x_0, n, W)

        if print_in_console:
            self.graph.print(True)
        # end of 1.A

        self.nodes.append(n)

    def predict(self, u, dt=None):
        # 1.B
        if print_in_console:
            print('#####pred#######')

        x = np.zeros(3)
        n = self.graph.add_node_pose_2d(x)

        G, V = state_jacobian(self.mu, u)
        R = get_motion_noise_covariance(u, self.alphas)

        W_u = inv(V@R@V.T)  # V allows to avoid spike in error graph

        if print_in_console:
            print(self.graph.get_estimated_state())

        self.graph.add_factor_2poses_2d_odom(u, self.nodes[-1], n, W_u)
        mu = self.graph.get_estimated_state()

        if print_in_console:
            print(mu)

        # end of 1.B
        # 2.D
        self.state.Sigma = V@R@V.T
        # end of 2.D
        self.nodes.append(n)

    def update(self, z):
        # 1.C
        if print_in_console:
            print('#####update#######')

        x, y, theta = self.mu

        for i in range(z.shape[0]):
            if z[i, 2] not in self.observed_lms:            # new landmark
                n = self.graph.add_node_landmark_2d(np.zeros(2))

                self.observed_lms[z[i, 2]] = n
                init = True

            else:
                n = self.observed_lms[z[i, 2]]

                init = False

            W_z = inv(self.Q)
            self.graph.add_factor_1pose_1landmark_2d(z[i, :2], self.nodes[-1], n, W_z, initializeLandmark=init)

        if print_in_console:
            print(self.graph.get_estimated_state())
        # end of 1.C

    def solve(self):
        if not batch_solution:
            # 1.D
            if print_in_console:
                print('#####solve#######')
            self.graph.solve(mrob.GN) #self.graph.solve(mrob.LM) #


        # 2.A
        self.errors.append(self.graph.chi2())
        # end of 2.A

        states = self.graph.get_estimated_state()
        if print_in_console:
            print(states)
        #end of 1.D
        self.state.mu = states[self.nodes[-1]]

    def solve_last(self, method='gn'):
        #2.E
        if batch_solution:
            if method == 'gn':
                alg = mrob.GN
            else:
                alg = mrob.LM
            prev_chi = self.graph.chi2()
            num_it = 0
            while(True):
                num_it += 1
                self.graph.solve(alg)
                cur_chi = self.graph.chi2()
                if (prev_chi - cur_chi) < 0.0001:
                    break
                prev_chi = cur_chi

            print(f'number of iterations: {num_it}')
        print(f'achived chi: {self.graph.chi2()}')
        # end of 2.E

    def graph_viz(self, last=False):
        # 2.B
        trajectory = np.zeros((2, len(self.nodes)))
        s = np.array(self.graph.get_estimated_state())
        for i, n in enumerate(self.nodes):
            trajectory[0, i] = s[n][0]
            trajectory[1, i] = s[n][1]
        # landmarks
        if not batch_solution or last:
            for p in self.observed_lms.values():
                plt.plot(s[p][0], s[p][1], 'ob')

        # pose
        plt.plot(s[self.nodes[-1]][0], s[self.nodes[-1]][1], 'oy')

        # trajectory
        trajectory = np.array(trajectory)
        plt.plot(trajectory[0], trajectory[1])

        # sigma
        # 2.D
        if last:
            mu = s[self.nodes[-1]][:2].reshape(2)
            sigma = self.Sigma[:2, :2]
            print(sigma)
            plot2dcov(mu, sigma)
        # end of 2.D

        # end of 2.B

    def plot_errors(self, dt):
        # 2.A
        time = np.arange(0, len(self.errors) * dt, dt)
        plt.figure(figsize=(12, 12))
        plt.plot(time, self.errors, color='b')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel(r'$\chi^2$', fontsize=16)
        plt.xlabel('time', fontsize=16)
        plt.title('', fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
        # end of 2.A

    def matrices(self):
        # 2.C
        print(f'adj. matrix: {self.graph.get_adjacency_matrix().shape}')
        print(f'cov. matrix: {self.graph.get_information_matrix().shape}')
        plt.figure(figsize=(12, 12))
        plt.spy(self.graph.get_adjacency_matrix(), marker='o', markersize=1)
        plt.title('Adjacency matrix $A$')

        plt.figure(figsize=(12, 12))
        plt.spy(self.graph.get_information_matrix(), marker='o', markersize=1)
        plt.title('Information matrix')
        # end of 2.C

        # 2.D
        plt.figure(figsize=(12, 12))
        plt.spy(inv(self.graph.get_information_matrix().toarray()), marker='o', markersize=1)
        plt.title('Covariance matrix')
        # end of 2.D

    @property
    def mu(self):
        """
        :return: The state mean after the update step (format: 1D array for easy indexing).
        """
        return self.state.mu.T[0]

    @property
    def Sigma(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self.state.Sigma
