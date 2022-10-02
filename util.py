import numpy as np
def rotate2D(x, angle):
  rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

  out = np.matmul(rot, x)
  return out.squeeze(-1)

def calculate_degree(initial, angular_vel, delta_time):
  angle = initial + np.rad2deg(angular_vel) * delta_time
  return angle

def calculate_pos(initial, velocity, delta_time, angle = 0.0):
  out = initial + rotate2D(velocity, np.deg2rad(-angle)) * delta_time 
  return out

class Dectected():
  global_id = 0
  def __init__(self, x, y):
    self.id += self.global_id
    self.global_id += 1
    self.x = x
    self.y = y
  def update(self, x, y):
    self.x = x
    self.y = y

class TempDetected():
  def __init__(self, x, y, id):
    self.x = x
    self.y = y
    self.id = id

def distance(x,y,x1,y1):
  return np.sqrt(np.square(y-x) + np.square(y1-x1) + 1e-6)
