import numpy as np
class KalmanFilter():
  def __init__(self, dt=0.1, u_x=1, u_y=1,
               std_acc=1, x_std_meas=0.1, y_std_meas=0.1):
    """
    :param dt: sampling time (time for 1 cycle)
    :param u_x: acceleration in x-direction
    :param u_y: acceleration in y-direction
    :param std_acc: process noise magnitude
    :param x_std_meas: standard deviation of the measurement in x-direction
    :param y_std_meas: standard deviation of the measurement in y-direction
    """

    self.dt = dt                              # Define sampling time
    self.u = np.matrix([[u_x],[u_y]])         # Define the  control input variables
    self.x = np.matrix([[0], [0], [0], [0]])  # Initial State

    # Define the State Transition Matrix A
    self.A = np.matrix([[1, 0, self.dt, 0],
                        [0, 1, 0, self.dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

    # Define the Control Input Matrix B
    self.B = np.matrix([[(self.dt**2)/2, 0],
                        [0,(self.dt**2)/2],
                        [self.dt,0],
                        [0,self.dt]])

    # Define Measurement Mapping Matrix
    self.H = np.matrix([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

    #Initial Process Noise Covariance
    self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                        [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                        [(self.dt**3)/2, 0, self.dt**2, 0],
                        [0, (self.dt**3)/2, 0, self.dt**2]]) * std_acc**2

    #Initial Measurement Noise Covariance
    self.R = np.matrix([[x_std_meas**2,0],
                       [0, y_std_meas**2]])

    #Initial Covariance Matrix
    self.P = np.eye(self.A.shape[1])

  def predict(self):
    """
    :return: numpy matrix
    """
    # Update time state
    self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
    # Calculate error covariance
    self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
    return self.x[0:2]

  def update(self, z):
    """
    :param z: np.array
    :return: numpy matrix
    """
    # S = H*P*H'+R
    S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
    # Calculate the Kalman Gain
    # K = P * H'* inv(H*P*H'+R)
    K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
    self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
    I = np.eye(self.H.shape[1])
    # Update error covariance matrix
    self.P = (I - (K * self.H)) * self.P
    return self.x[0:2]