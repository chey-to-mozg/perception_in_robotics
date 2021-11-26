"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task
        self._state_bar.mu = np.expand_dims(get_prediction(self.mu, u), -1)
        G = self.G_t(u)
        V = self.V_t(u)
        R = get_motion_noise_covariance(u, self._alphas)
        self._state_bar.Sigma = G @ self.Sigma @ G.T + V @ R @ V.T

    def update(self, z):
        # TODO implement correction step
        bearing, lm_id = z[0], int(z[1])

        H = self.H_t(lm_id)

        K_t = self.Sigma_bar @ H.T / (H @ self.Sigma_bar @ H.T + self._Q) ## a^{-1} for scalars = 1/a

        h = get_expected_observation(self.mu_bar, lm_id)[0]

        self._state.mu = np.expand_dims(self.mu_bar + K_t * wrap_angle(bearing - h), -1)
        self._state.Sigma = (np.eye(K_t.shape[0]) - np.outer(K_t, H)) @ self.Sigma_bar

    def G_t(self,  motion):
        state = self.mu

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)

        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        G = np.array([
            [1, 0, -dtran * np.sin(theta + drot1)],
            [0, 1, dtran * np.cos(theta + drot1)],
            [0, 0, 1]
        ])

        return G

    def V_t(self,  motion):
        state = self.mu

        assert isinstance(state, np.ndarray)
        assert isinstance(motion, np.ndarray)

        assert state.shape == (3,)
        assert motion.shape == (3,)

        x, y, theta = state
        drot1, dtran, drot2 = motion

        V = np.array([
            [-dtran * np.sin(theta + drot1), np.cos(theta + drot1), 0],
            [dtran * np.cos(theta + drot1), np.sin(theta + drot1), 0],
            [1, 0, 1]
        ])

        return V

    def H_t(self,  lm_id):
        state = self.mu_bar

        m_x = self._field_map.landmarks_poses_x[lm_id]
        m_y = self._field_map.landmarks_poses_y[lm_id]

        assert isinstance(state, np.ndarray)

        assert state.shape == (3,)

        x, y, theta = state

        H = np.array([
            (m_y - y) / ((m_x - x) ** 2 + (m_y - y) ** 2),
            -(m_x - x) / ((m_x - x) ** 2 + (m_y - y) ** 2),
            -1
        ])

        return H
