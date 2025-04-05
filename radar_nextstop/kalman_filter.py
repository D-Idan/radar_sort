# kalman_filter.py

import numpy as np
from filterpy.kalman import KalmanFilter

class RadarKalmanFilter:
    """
    Kalman filter for radar tracking.
    State vector: [cx, cy, cz, vx, vy, vz, length, width, height, orientation]
    Measurement vector: [cx, cy, cz, length, width, height, orientation]
    """
    def __init__(self, dt=0.1):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        # State transition matrix
        self.kf.F = np.eye(10)
        self.kf.F[0, 3] = dt  # cx update via vx
        self.kf.F[1, 4] = dt  # cy update via vy
        self.kf.F[2, 5] = dt  # cz update via vz
        # Measurement matrix maps state to measurement space
        self.kf.H = np.zeros((7, 10))
        self.kf.H[0, 0] = 1.0  # cx
        self.kf.H[1, 1] = 1.0  # cy
        self.kf.H[2, 2] = 1.0  # cz
        self.kf.H[3, 6] = 1.0  # length
        self.kf.H[4, 7] = 1.0  # width
        self.kf.H[5, 8] = 1.0  # height
        self.kf.H[6, 9] = 1.0  # orientation
        # Covariances (tune these matrices as needed)
        self.kf.Q = np.eye(10) * 0.1  # process noise
        self.kf.R = np.eye(7) * 0.1   # measurement noise
        self.kf.P = np.eye(10) * 1.0  # initial uncertainty

    def initialize(self, initial_state):
        self.kf.x = np.array(initial_state).reshape(-1, 1)

    def predict(self):
        self.kf.predict()

    def update(self, measurement):
        self.kf.update(np.array(measurement).reshape(-1, 1))

    def get_state(self):
        return self.kf.x.flatten()
