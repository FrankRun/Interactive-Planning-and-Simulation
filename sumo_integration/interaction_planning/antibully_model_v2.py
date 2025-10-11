import time
import math
import sys
import numpy as np
import ctypes
sys.path.append('/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/sumo_integration/interaction_planning/')
import utils_v2

ll = ctypes.cdll.LoadLibrary

class Antibully_ACC:
    def __init__(self,world):
        '''CARLA API'''
        self.world = world
        '''必要参数'''
        self.length = 4
        self.timestep = 0.1
        self.headway = 0.5
        self.delta_t = 1.1
        self.horizon = 10
        self.pv_cv_threshold = 1.2
        self.x_des = -25
        self.v_des = 20
        self.speed_limit = 30
        '''风格辨识'''
        self.identification_No = 0
        self.lon_G0 = np.eye(6)
        self.lat_G0 = np.eye(4)
        self.lon_Q0 = np.zeros((6,6))
        self.lat_Q0 = np.zeros((4,4))
        self.lon_theta0 = np.array([[0],[0],[0]])
        self.lat_theta0 = np.array([[0],[0]])
        self.lon_beta = np.array([[0],[0],[0]])
        self.lat_beta = np.array([[0],[0]])
        self.lon_error = 0.01
        self.lat_error = 0.001
        self.ego_attributes = []
        self.all_veh_attributes = []
        '''切入前车'''
        self.cutin_vehicle = {"No":0, "X_distance":0, "Y_distance":0, "lon_speed":0, "lon_a":0, 
                              "lat_speed":0, "efi":0, "afi":0, "if_pv":0, "if_cv":0}
        '''主车本车道前车'''
        self.pre_vehicle = {"No":0, "X_distance":0, "lon_speed":0, "lon_a":0}

    def data_predeal(self,ego_vehicle):
        
        if ego_vehicle.FV[0] != False and ego_vehicle.FV[0] != '':
            '''本车道前车存在'''
            self.pre_vehicle['No'] = ego_vehicle.FV[0]
            self.pre_vehicle['X_distance'] = -ego_vehicle.FV[1]
            self.pre_vehicle['lon_speed'] = ego_vehicle.FV[2]
            self.pre_vehicle['lon_a'] = ego_vehicle.FV[4]
            '''可能切入车存在'''
            if ego_vehicle.LFV[0] != False and ego_vehicle.LFV[0] != '':
                if ego_vehicle.LFV[1] > ego_vehicle.FV[1]:
                    self.cutin_vehicle = {"No":0, "X_distance":0, "Y_distance":0, "lon_speed":0, "lon_a":0, 
                                     "lat_speed":0, "efi":0, "afi":0, "if_pv":0, "if_cv":0}
                else:
                    self.cutin_vehicle['No'] = ego_vehicle.LFV[0]
                    self.cutin_vehicle['X_distance'] = -ego_vehicle.LFV[1]
                    self.cutin_vehicle['Y_distance'] = ego_vehicle.y-ego_vehicle.LFV[3]
                    self.cutin_vehicle['lon_speed'] = ego_vehicle.LFV[2]
                    self.cutin_vehicle['lon_a'] = -ego_vehicle.LFV[4]
                    self.cutin_vehicle['if_cv'] = 1
            else:
                self.cutin_vehicle = {"No":0, "X_distance":0, "Y_distance":0, "lon_speed":0, "lon_a":0, 
                                     "lat_speed":0, "efi":0, "afi":0, "if_pv":0, "if_cv":0}
        else:
            self.pre_vehicle = {"No":0, "X_distance":0, "lon_speed":0, "lon_a":0}
            '''可能切入车存在'''
            if ego_vehicle.LFV[0] != False and ego_vehicle.LFV[0] != '':
                self.cutin_vehicle['No'] = ego_vehicle.LFV[0]
                self.cutin_vehicle['X_distance'] = -ego_vehicle.LFV[1]
                self.cutin_vehicle['Y_distance'] = ego_vehicle.y-ego_vehicle.LFV[3]
                self.cutin_vehicle['lon_speed'] = ego_vehicle.LFV[2]
                self.cutin_vehicle['lon_a'] = -ego_vehicle.LFV[4]
                self.cutin_vehicle['if_cv'] = 1
            else:
                self.cutin_vehicle = {"No":0, "X_distance":0, "Y_distance":0, "lon_speed":0, "lon_a":0, 
                                     "lat_speed":0, "efi":0, "afi":0, "if_pv":0, "if_cv":0}
        return self.cutin_vehicle, self.pre_vehicle
    
    def behaviour_identification(self,ego_vehicle):
        '''是否进入驾驶风格辨识'''
        if self.cutin_vehicle["No"] == 0: 
            self.identification_No = 0
            self.lon_theta0 = np.array([[0],[0],[0]])
            self.lat_theta0 = np.array([[0],[0]])
            self.lon_beta = np.array([[0],[0],[0]])
            self.lat_beta = np.array([[0],[0]])
            return 0,0,0
            
        if self.identification_No != self.cutin_vehicle["No"]:
            self.identification_No = 0
            self.lon_theta0 = np.array([[0],[0],[0]])
            self.lat_theta0 = np.array([[0],[0]])
            self.lon_beta = np.array([[0],[0],[0]])
            self.lat_beta = np.array([[0],[0]])
        
        '''进入纵向风格辨识'''
            
        lon_f_xdot = np.transpose(np.array([[1,self.timestep,-self.timestep],
                                            [0,1,0],
                                            [0,0,1]])) #3X3 A_k
        lon_f_udot = np.array([[-0.5*self.timestep**2,0,self.timestep]]) #1X3 C_k的转置
        
        lon_l_xdot = np.array([[2*(self.cutin_vehicle["X_distance"]-self.x_des),0,0],
                                [0,0,0],
                                [0,2*(self.cutin_vehicle["lon_speed"]-self.v_des),0]]) #3X3 L_k对X_k求梯度
        
        lon_l_udot = np.array([[0,0,2*self.cutin_vehicle["lon_a"]]])  #1X3 L_k对a_k求梯度
        
        lon_F = np.hstack((lon_l_udot,lon_f_udot)) #1X6 花体L_k
        
        lon_G1 = np.hstack((np.eye(3),np.zeros((3,3)))) #花体M_k 上半部分
        lon_G2_1 = np.dot(-np.linalg.inv(lon_f_xdot),lon_l_xdot) #3X3
        lon_G2_2 = np.linalg.inv(lon_f_xdot) #3X3
        lon_G2 = np.hstack((lon_G2_1,lon_G2_2))  #3X6 花体M_k 下半部分
        lon_G = np.dot(np.vstack((lon_G1,lon_G2)) , self.lon_Q0) #6X6 花体M_k
        
        lon_Q = np.dot(np.transpose(np.dot(lon_F,lon_G)) , np.dot(lon_F,lon_G)) + self.lon_Q0  #6X6 Q_k
        lon_Q_main = lon_Q[1:,1:] #5X5
        lon_q = lon_Q[1:,0] #5X1 q_k
        lon_alpha1 = -np.dot(np.linalg.pinv(lon_Q_main),lon_q.reshape(-1,1)) #5X1
        lon_alpha = np.vstack((np.array([[1]]),lon_alpha1)) #6X1 Z_0
        lon_theta = np.dot(lon_G1,lon_alpha) #3X1 β
        
        if np.linalg.norm(lon_theta-self.lon_theta0,2) < self.lon_error:
            self.lon_theta0 = np.array([[0],[0],[0]])
            #self.lon_theta0 = lon_theta
            self.lon_beta = lon_theta
            self.lon_G0 = np.eye(6)
            self.lon_Q0 = np.zeros((6,6))
        else:
            self.lon_theta0 = lon_theta
            self.lon_G0 = lon_G
            self.lon_Q0 = lon_Q
        
        '''进入横向参数辨识'''
        v_cv = self.cutin_vehicle["lon_speed"]
        lat_f_xdot = np.transpose(np.array([[0,1],[1,v_cv*self.timestep]])) #2X2
        lat_f_udot = np.array([[0.25*v_cv*self.timestep,0]]) #1X2  C_T
        lat_l_udot = np.array([[0,0]])  #1X2
        lat_l_xdot = np.array([[2*(self.cutin_vehicle["efi"]),0],
                                [0,-2*(self.cutin_vehicle["Y_distance"])]])   #2X2
        
        lat_F = np.hstack((lat_l_udot,lat_f_udot)) #1X4
        #print("lon_F",lon_F.shape)
        lat_G1 = np.hstack((np.eye(2),np.zeros((2,2))))
        lat_G2_1 = np.dot(-np.linalg.inv(lat_f_xdot),lat_l_xdot) #2X2
        lat_G2_2 = np.linalg.inv(lat_f_xdot) #2X2
        lat_G2 = np.hstack((lat_G2_1,lat_G2_2))  #2X4
        lat_G = np.vstack((lat_G1,lat_G2)) #4X4
        lat_G = np.dot(lat_G,self.lat_G0)
        #print("lon_G",lon_G.shape)
        lat_Q = np.dot(np.transpose(np.dot(lat_F,lat_G)) , np.dot(lat_F,lat_G)) + self.lat_Q0  #4X4
        #print("lon_Q",lon_Q.shape)
        lat_Q_main = lat_Q[1:,1:] #3X3
        #print("lon_Q_main",lon_Q_main.shape)
        lat_q = lat_Q[1:,0] #3X1
        #print("lon_q",lon_q.reshape(-1,1).shape)
        lat_alpha1 = -np.dot(np.linalg.pinv(lat_Q_main),lat_q.reshape(-1,1))
        #print("lon_alpha1",lon_alpha1.shape)
        lat_alpha = np.vstack((np.array([[1]]),lat_alpha1)) #4X1
        #print("lon_alpha",lon_alpha.shape)
        lat_theta = np.dot(lat_G1,lat_alpha) #2X1
        
        if np.linalg.norm(lat_theta-self.lat_theta0,2) < self.lat_error:
            self.lat_theta0 = np.array([[0],[0]])
            self.lat_beta = lat_theta
            self.lat_G0 = np.eye(4)
            self.lat_Q0 = np.zeros((4,4))
        else:
            self.lat_theta0 = lat_theta
            self.lat_G0 = lat_G
            self.lat_Q0 = lat_Q

        self.identification_No = self.cutin_vehicle["No"]
        return self.identification_No,self.lon_beta,self.lat_beta
    
    def ACC(self,ego_vehicle):
        flag = 1
        y_FV = 1000; y_LFV = 1000
        '''当前车道前车是否存在'''
        if ego_vehicle.FV[0] == False:
            # 当前车道前车不存在
            pass
        else:
            y_FV = ego_vehicle.FV[3]
        '''左前车是否存在'''
        if ego_vehicle.LFV[0] == False:
            # 左前车不存在
            pass
        else:
            y_LFV = ego_vehicle.LFV[3]
        
        '''判断前车是哪个'''
        if abs(ego_vehicle.y - y_FV) <= 1:
            # 前车是FV
            x_pv = ego_vehicle.FV[1] + ego_vehicle.x
            y_pv = y_FV
            v_pv = ego_vehicle.FV[2]
        else:
            if abs(ego_vehicle.y - y_LFV) <= 1:
                # 前车是LFV
                x_pv = ego_vehicle.LFV[1] + ego_vehicle.x
                y_pv = y_LFV
                v_pv = ego_vehicle.FV[2]
            else:
                flag = 0
                #print("不存在前车")
                return flag, 0 , 0
        THW = (x_pv - ego_vehicle.x)/(ego_vehicle.speed - v_pv)
        '''ACC planner'''
        if x_pv - ego_vehicle.x < 10:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading,
                                        x_pv, -20, v_pv, y_pv, self.timestep, 10,
                                        10, 1, 20, 5, 150)
            #print("距离小于10, 安全因素强制开启ACC")
            return flag, u, f
        if THW <= 0 :
            flag = 0
            #print("THW小于0，无需开启ACC")
            return flag, 0 , 0
        elif THW <= 2:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading,
                                        x_pv, -30, v_pv, y_pv, self.timestep, 10,
                                        10, 10, 30, 5, 150)
            #print("THW小于2, 安全因素开启ACC")
            return flag, u, f
        elif THW <= 5:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading,
                                        x_pv, -20, v_pv, y_pv, self.timestep, 10,
                                        10, 10, 50, 5, 150)
            #print("THW小于5, 安全因素开启ACC")
            return flag, u, f
        else:
            flag = 0
            #print("存在前车但无需开启ACC")
            return flag, 0 , 0
    
    def antibully_acc(self,ego_vehicle):
        dis_des = 0
        lib_anti = ll("/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/sumo_integration/interaction_planning/controller/antibully_acc.dll")
        lib_anti.antibully_acc.restype = ctypes.c_double
        u = lib_anti.antibully_acc(ctypes.c_double(1),  #Ti
                                   ctypes.c_double(self.timestep), #step
                                   ctypes.c_double(11),  #N
                                   ctypes.c_double(float(self.cutin_vehicle["X_distance"])), #delta_x
                                   ctypes.c_double(ego_vehicle.speed), #v_ev
                                   ctypes.c_double(self.cutin_vehicle["lon_speed"]), #v_cv
                                   ctypes.c_double(ego_vehicle.LFV[3]), #y_cv
                                   ctypes.c_double(float(self.cutin_vehicle['efi'])), #efi
                                   ctypes.c_double(dis_des), #防霸凌期望间距
                                   ctypes.c_double(self.x_des), #对手车期望间距
                                   ctypes.c_double(self.v_des), #防霸凌期望速度
                                   ctypes.c_double(self.v_des), #对手车期望速度
                                   ctypes.c_double(ego_vehicle.y), #对手车期望车道
                                   ctypes.c_double(0), #对手车期望航向角
                                   ctypes.c_double(self.lon_beta[0,0]), #纵向辨识参数
                                   ctypes.c_double(self.lon_beta[1,0]), #纵向辨识参数
                                   ctypes.c_double(self.lon_beta[2,0]), #纵向辨识参数
                                   ctypes.c_double(self.lat_beta[0,0]), #横向辨识参数
                                   ctypes.c_double(self.lat_beta[1,0]), #横向辨识参数
                                   ctypes.c_double(10), #自车参数
                                   ctypes.c_double(10), #自车参数
                                   ctypes.c_double(100), #自车参数
                                   ctypes.c_double(self.speed_limit)  #道路限速
                                   )
        
        return u,0
    
    
    def planning(self,ego_vehicle):
        '''ACC activated'''
        flag, u, f = self.ACC(ego_vehicle)
        if flag == 0:
            pass
        else:
            return u, f
        '''博弈控制执行'''
        if (self.lon_beta == np.array([[0],[0],[0]])).all() or (self.lat_beta == np.array([[0],[0]])).all():
            #print("不开启博弈式控制，开启巡航")
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 40, -25, 20, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        1, 25, 1, 50, 100)
        else:
            #print("开启博弈式控制")
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x-self.cutin_vehicle["X_distance"], 0, 25, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        10, 10, 10, 50, 100)
            
        if u > 3.5:
            u = 3.5
        if u < -4:
            u = -4
        if np.isnan(u):
            u = 0
        return u,f
