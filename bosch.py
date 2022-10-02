import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from util import *
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
import uuid
from model import *
from dataclasses import dataclass

@dataclass
class Tracker:
  id: int
  centroid: tuple
  kalman: KalmanFilter()
  consecutive_invi_num = 0

#Extracting car information from four files
"""
vxvRef: longi velocity
axvRef: longi acceleration
vyvRef: lateral velocity
ayvRef: lateral acceleration
psiDtOpt: angular speed around axis z (yaw rate)
"""


#getting first 4 directories to use as training set
dataset_dir  = os.listdir('dataset')
train_dataset_dir = dataset_dir[:-1]
val_dataset_dir = dataset_dir[-1]

#car dataset
car_accelerationX = []
car_accelerationY = []
car_velocityX = []
car_velocityY = []
car_angularSpeed = []
car_time = []

#loop through the directories and gather information on the motor data
for dir in train_dataset_dir:
    car_data = pd.read_csv(os.path.join(os.getcwd(), 'dataset', dir, f'Group_416.csv'))

    car_accelerationX.append(car_data[['_g_ods_OneDrivingSW_perHv_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._1_._elem._axvRef_sw']].to_numpy()/2048)
    car_accelerationY.append(car_data[['_g_ods_OneDrivingSW_perHv_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._1_._elem._ayvRef_sw']].to_numpy()/2048)
    car_velocityX.append(car_data[['_g_ods_OneDrivingSW_perHv_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._1_._elem._vxvRef_sw']].to_numpy()/256)
    car_velocityY.append(car_data[['_g_ods_OneDrivingSW_perHv_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._1_._elem._vyvRef_sw']].to_numpy()/256)
    car_angularSpeed.append(car_data[['_g_ods_OneDrivingSW_perHv_HV_PerPmeRunnable_PerPmeRunnable_m_pmePort_out_local.TChangeableMemPool._._._m_arrayPool._1_._elem._psiDtOpt_sw']].to_numpy()/16384)
    car_time.append(car_data[['t']].to_numpy())

#merging of all four train dataset
# car_accelerationX = np.concatenate(car_accelerationX, axis = 0)
# car_accelerationY = np.concatenate(car_accelerationY, axis = 0)
# car_velocityX = np.concatenate(car_velocityX, axis = 0)
# car_velocityY = np.concatenate(car_velocityY, axis = 0)
# car_angularSpeed = np.concatenate(car_angularSpeed, axis = 0)
# car_time = np.concatenate(car_time, axis = 0)
car_positionX = []
car_positionY = []
car_angle = []

#calculate position of reference vehicle using simple kinematics formula
for sample in range(len(car_time)):
    car_px_temp = []
    car_py_temp = []
    car_angle_temp = []
    for trajectory in range(len(car_time[sample])):
        if(trajectory == 0):
            posX, posY = calculate_pos([0.0, 0.0], np.array([car_velocityX[sample][trajectory], car_velocityY[sample][trajectory]]).reshape(-1, 1), car_time[sample][trajectory])
            angle = calculate_degree(0.0, car_angularSpeed[sample][trajectory], car_time[sample][trajectory])[0]
            
        else:
            posX, posY = calculate_pos([car_px_temp[-1], car_py_temp[-1]], np.array([car_velocityX[sample][trajectory], car_velocityY[sample][trajectory]]).reshape(-1, 1), car_time[sample][trajectory] - car_time[sample][trajectory-1], angle = car_angle_temp[-1])
            angle = calculate_degree(car_angle_temp[-1], car_angularSpeed[sample][trajectory], car_time[sample][trajectory] - car_time[sample][trajectory-1])[0]

        car_px_temp.append(posX)
        car_py_temp.append(posY)
        car_angle_temp.append(angle)
    car_positionX.append(car_px_temp)
    car_positionY.append(car_py_temp)
    car_angle.append(car_angle_temp)

#Get the sensor data
all_sensor_data = []
for dir in train_dataset_dir:
    sensor_data = pd.read_csv(os.path.join(os.getcwd(), 'dataset', dir, f'Group_349.csv'))
    all_sensor_data.append(sensor_data)

#Get the camera data
all_camera_data = []
all_radar_data = []
for sense_data in all_sensor_data:
    camera_data = []
    num_camera_dect = 15
    for item in range(num_camera_dect):
        obj_cam_data = sense_data[['t', f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_camData._m_objects._m_value._{item}_._m_dx',
        f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_camData._m_objects._m_value._{item}_._m_dy',
        f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_camData._m_objects._m_value._{item}_._m_vx',
        f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_camData._m_objects._m_value._{item}_._m_vy'
        ]].to_numpy()
        camera_data.append(obj_cam_data)
    all_camera_data.append(camera_data)

    #Get the radar data
    

    radar_data = {}
    num_radar_sensor = 4
    for item in range(num_radar_sensor):
        radar_data[f"radar_sensor_{item}"] = []
        num_radar_dect = 10
        for item_dect in range(num_radar_dect):
            obj_cam_data = sense_data[['t', f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_cornerData._m_value._{item}_._m_objects._m_value._{item_dect}_._m_dx',
            f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_cornerData._m_value._{item}_._m_objects._m_value._{item_dect}_._m_dy',
            f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_cornerData._m_value._{item}_._m_objects._m_value._{item_dect}_._m_vx',
            f'_g_Infrastructure_CCR_NET_NetRunnablesClass_m_rteInputData_out_local.TChangeableMemPool._._._m_arrayPool._0_._elem._m_cornerData._m_value._{item}_._m_objects._m_value._{item_dect}_._m_vy']].to_numpy()
            radar_data[f"radar_sensor_{item}"].append(obj_cam_data)
    all_radar_data.append(radar_data)

#Get complete data of a sample
all_obj_1 = []
all_obj_2 = []
for dir in train_dataset_dir:
    obj_1 = pd.read_csv(os.path.join(os.getcwd(), 'dataset', dir, f'Group_343.csv')).to_numpy()
    obj_2 = pd.read_csv(os.path.join(os.getcwd(), 'dataset', dir, f'Group_342.csv')).to_numpy()
    all_obj_1.append(obj_1)
    all_obj_2.append(obj_2)

num_frames = 1000
prev_obj = []
curr_obj = []

tracking_obj = {}

#Performing kalman filter
x = np.matrix('0. 0. 0. 0.').T 
P = np.matrix(np.eye(4))*1000 # initial uncertainty
R = 0.01**2

#Visualize moving car
fig, ax = plt.subplots(nrows=3, ncols=1, figsize= (14,14))
size = 2


def generate_animation(i):
    for j in range(3):
        ax[j].clear()
        global curr_obj, prev_obj, car_positionX, car_positionY, car_angle, camera_data, all_camera_data, all_obj_2, all_obj_1, all_radar_data
        global x, P, R

        ax[j].add_patch(Rectangle((car_positionX[j][i], car_positionY[j][i]), size, size, color='r'))
        ax[j].text(car_positionX[j][i], car_positionY[j][i], f"reference car")

        strt_idx = int(len(car_positionY[j]) - len(all_obj_1[j]))//2
            
        ax[j].add_patch(Rectangle((car_positionX[j][i] - all_obj_1[j][(strt_idx+i) * 2][1], car_positionY[j][i] -  all_obj_1[j][(strt_idx+i) * 2][2]), size, size, color='y'))
        ax[j].text(car_positionX[j][i] - all_obj_1[j][(strt_idx+i) * 2][1], car_positionY[j][i] -  all_obj_1[j][(strt_idx+i) * 2][2], f"detection_gps_1")

        ax[j].add_patch(Rectangle((car_positionX[j][i] - all_obj_2[j][(strt_idx+i) * 2][1], car_positionY[j][i] - all_obj_2[j][(strt_idx+i) * 2][2]), size, size, color='y'))
        ax[j].text(car_positionX[j][i] - all_obj_2[j][(strt_idx+i) * 2][1], car_positionY[j][i] - all_obj_2[j][(strt_idx+i) * 2][2], f"detection_gps_2")
        
        ### Using the hungarian algorithm and using the l2-norm as my cost
        for index in range(num_camera_dect):
            cam_x = (all_camera_data[j][index][i*2][1]/128.0) - car_positionX[j][i]
            cam_y = (all_camera_data[j][index][i*2][2]/128.0) - car_positionY[j][i]
            curr_obj.append([cam_x, cam_y, str(uuid.uuid4())[:5]])

        for _, value in all_radar_data[j].items():
            for index in range(num_radar_dect):
                cam_x = car_positionX[j][i] - (value[index][i*2][1]/128.0)
                cam_y = car_positionY[j][i] - (value[index][i*2][2]/128.0)

                curr_obj.append([cam_x, cam_y, str(uuid.uuid4())[:5]])
        
        if(len(prev_obj) <= 0):
            prev_obj = curr_obj
            curr_obj = []
            
        else:
            score_matrix = np.zeros((len(curr_obj), len(prev_obj)))
            for t in range(len(curr_obj)):
                for x in range(len(prev_obj)):
                    score_matrix[t, x] = distance(curr_obj[t][0],curr_obj[t][1],prev_obj[t][0],prev_obj[t][1])
                
            row_ind, col_ind = linear_sum_assignment(score_matrix)

            for r, c in zip(row_ind, col_ind):
                curr_obj[r][2] = prev_obj[c][2]

            prev_obj = curr_obj
            curr_obj = []

        for index in range(len(prev_obj)):
            cam_x = prev_obj[index][0]
            cam_y = prev_obj[index][1]
            #Indicator
            close_x = (cam_x - car_positionX[j][i] + 0.25)**2
            close_y = (cam_y - car_positionY[j][i] + 2)**2
            if(close_x < 0.1 and close_y < 0.02):
                ax[j].scatter([car_positionX[j][i]], [car_positionY[j][i]], color='r', s=200)
            # #kalman estimation
            # result = []
            # x, P = kalman_xy(x, P, (cam_x, cam_y), R)
            # result.append((x[:2]).tolist())
            # kal_x, kal_y = zip(*result)

            #ax[j].scatter(kal_x, kal_y)
            ax[j].add_patch(Rectangle((cam_x, cam_y), size, size))
            ax[j].text(cam_x, cam_y, prev_obj[index][2])


        ax[j].set_xlim(car_positionX[j][i]-25, car_positionX[j][i]+25)
        ax[j].set_ylim(car_positionY[j][i]-25, car_positionY[j][i]+25)
        ax[j].set_xlabel('X(m)')
        ax[j].set_ylabel('Y(m)')
        ax[j].set_title(f'car detection from dataset_{j}')
    
animation = FuncAnimation(fig, generate_animation, 
                          frames = num_frames, interval = 40)

animation.save('car.gif')