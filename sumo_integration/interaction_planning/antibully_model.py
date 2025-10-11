import time
# from sumo_integration.interaction_planning.model import Res_DIRL
import sys
sys.path.append('/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/sumo_integration/interaction_planning/')
from model import Res_DIRL
import utils_v2
import math
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing


class AntiBully():
    def __init__(self,world):
        '''CARLA API'''
        self.world = world
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''必要参数'''
        self.length = 4
        self.timestep = 0.1
        self.headway = 0.5
        self.delta_t = 1.1
        self.horizon = 10
        self.pv_cv_threshold = 1.2
        self.x_des = 30
        self.v_des = 15
        self.speed_limit = 30
        '''决策模型'''
        self.model = Res_DIRL()
        self.model = torch.load("/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/sumo_integration/interaction_planning/model1.pth")
        self.min_max_scaler = preprocessing.MinMaxScaler()
        '''风格辨识'''
        self.identification_No = 0
        self.lon_G0 = np.eye(6)
        self.lon_Q0 = np.zeros((6,6))
        self.lon_theta0 = np.array([[0],[0],[0]])
        self.lon_beta = np.array([[0],[0],[0]])
        self.lon_error = 0.1
        self.ego_attributes = []
        self.all_veh_attributes = []
        '''主车要加塞的车辆'''
        self.first_lead_vehicle = {"No":0, "X_distance":0, "Y_distance":0, "lon_speed":0, "lon_a":0}

    def data_update_left(self,ego_vehicle,time):
        '''导入SUMO采集的数据并处理'''
        data = [0, 0, 0, -100, -30, -100, -30, 100, 30, 100, 30, 100, 30]
        # 主车数据
        data[0] = ego_vehicle.speed
        data[1] = ego_vehicle.acceleration
        data[2] = ego_vehicle.x   ######need to be fixed
        # V1
        if ego_vehicle.LRRV[0] != False:
            if -ego_vehicle.LRRV[1] > -100:
                data[3] = -ego_vehicle.LRRV[1]
                data[4] = ego_vehicle.LRRV[2] - ego_vehicle.speed
        # V2
        if ego_vehicle.LRV[0] != False:
            if -ego_vehicle.LRV[1] > -100:
                data[5] = -ego_vehicle.LRV[1]
                data[6] = ego_vehicle.LRV[2] - ego_vehicle.speed
        # V3
        if ego_vehicle.LFV[0] != False:
            if ego_vehicle.LFV[1] < 100:
                data[7] = ego_vehicle.LFV[1]
                data[8] = ego_vehicle.LFV[2] - ego_vehicle.speed
        # V4
        if ego_vehicle.LFFV[0] != False:
            if ego_vehicle.LFFV[1] < 100:
                data[9] = ego_vehicle.LFFV[1]
                data[10] = ego_vehicle.LFFV[2] - ego_vehicle.speed
        # V5
        if ego_vehicle.FV[0] != False:
            if ego_vehicle.FV[1] < 100:
                data[11] = ego_vehicle.FV[1]
                data[12] = ego_vehicle.FV[2] - ego_vehicle.speed
        # DATA TRANSFORM
        data = pd.DataFrame(data).T
        return data
    
    def data_update_right(self,ego_vehicle,time):
        '''导入SUMO采集的数据并处理'''
        data = [0, 0, 0, -100, -30, -100, -30, 100, 30, 100, 30, 100, 30]
        # 主车数据
        data[0] = ego_vehicle.speed
        data[1] = ego_vehicle.acceleration
        data[2] = ego_vehicle.x   ######need to be fixed
        # V1
        if ego_vehicle.RRRV[0] != False and ego_vehicle.RRRV[0] != '':
            if -ego_vehicle.RRRV[1] > -100:
                data[3] = -ego_vehicle.LRRV[1]
                data[4] = ego_vehicle.LRRV[2] - ego_vehicle.speed
        # V2
        if ego_vehicle.RRV[0] != False and ego_vehicle.RRV[0] != '':
            if -ego_vehicle.RRV[1] > -100:
                data[5] = -ego_vehicle.LRV[1]
                data[6] = ego_vehicle.LRV[2] - ego_vehicle.speed
        # V3
        if ego_vehicle.RFV[0] != False and ego_vehicle.RFV[0] != '':
            if ego_vehicle.RFV[1] < 100:
                data[7] = ego_vehicle.LFV[1]
                data[8] = ego_vehicle.LFV[2] - ego_vehicle.speed
        # V4
        if ego_vehicle.RFFV[0] != False and ego_vehicle.RFFV[0] != '':
            if ego_vehicle.RFFV[1] < 100:
                data[9] = ego_vehicle.LFFV[1]
                data[10] = ego_vehicle.LFFV[2] - ego_vehicle.speed
        # V5
        if ego_vehicle.FV[0] != False and ego_vehicle.FV[0] != '':
            if ego_vehicle.FV[1] < 100:
                data[11] = ego_vehicle.FV[1]
                data[12] = ego_vehicle.FV[2] - ego_vehicle.speed
        # DATA TRANSFORM
        data = pd.DataFrame(data).T
        return data
    
    def decision_making(self,data, dataset):
        '''交互式决策算法'''
        new_dataset = dataset.loc[:,:12]
        new_data = pd.concat([data, new_dataset])
        new_data = new_data.reset_index(drop=True)
        new_data = self.min_max_scaler.fit_transform(new_data)
        data = new_data[0,:].reshape(1,-1)
        action = [0,0.25,0.5,0.75,1]
        data = torch.FloatTensor(data).to(self.device)
        action0 = action[0] * torch.ones((1, 1)).to(self.device)
        action1 = action[1] * torch.ones((1, 1)).to(self.device)
        action2 = action[2] * torch.ones((1, 1)).to(self.device)
        action3 = action[3] * torch.ones((1, 1)).to(self.device)
        action4 = action[4] * torch.ones((1, 1)).to(self.device)
        out0 = self.model(data,action0)
        out1 = self.model(data,action1)
        out2 = self.model(data,action2)
        out3 = self.model(data,action3)
        out4 = self.model(data,action4)
        #合并
        output = torch.cat([out0, out1, out2, out3, out4], dim=1).to("cpu")
        output = torch.argmax(output, dim=1).detach().numpy()
        return output[0]+1
    
    def select_competevehicle(self,ego_vehicle):
        '''决策加塞时选择博弈车'''
        # 博弈车对象
        compete_target = False
        if ego_vehicle.LRV[0] != False:
            # 博弈车存在且间距小于50
            if ego_vehicle.LRV[1] < 50:
                compete_target = ego_vehicle.LRV
                # 博弈车数据处理
                self.competevehicle_process(compete_target,ego_vehicle)
        return compete_target
    
    def competevehicle_process(self,compete_target,ego_vehicle):
        '''博弈车数据处理'''
        if compete_target == False:
            # 不存在博弈车
            pass
        else:
            self.first_lead_vehicle['No'] = compete_target[0]
            self.first_lead_vehicle["X_distance"] = compete_target[1]
            self.first_lead_vehicle["Y_distance"] = ego_vehicle.y - compete_target[3]
            self.first_lead_vehicle["lon_speed"] = compete_target[2]
            self.first_lead_vehicle["lon_a"] = compete_target[4]

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
        
        
    def OMPC(self, decision, ego_vehicle, y_des):
        # ACC activated
        flag, u, f = self.ACC(ego_vehicle)
        if flag == 0:
            pass
        else:
            # print("开启ACC:",u,f)
            return u, f
        # planner
        if decision == 1 or decision == 5:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x, -30, 10, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        1, 1, 100, 10, 100)
            # print("决策减速:",u,f)
        elif decision == 3 or decision == 4:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 60, -25, 25, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        10, 1, 10, 50, 100)
            #print("y_des:", ego_vehicle.speed,ego_vehicle.y, ego_vehicle.y - ego_vehicle.lateral_pos, ego_vehicle.heading, ego_vehicle.lane_heading)
            # print("决策加速:",u,f)
        elif decision == 2:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 40, -15, self.v_des, y_des, 
                                        self.timestep, 10, 1, 1, 10, 50, 200)
            # print("决策无博弈换道:",u,f)
        else:
            # print("未决策")
            u,f = 0 , 0
        
        if u > 3: u = 3
        elif u < -3:u = -3
        else: u = u
        
        if f > 0.7: f = 0.7
        elif f < -0.7:f = -0.7
        else: f = f

        return u,f

    def behaviour_identification(self):
        '''是否进入驾驶风格辨识'''
        if self.first_lead_vehicle["No"] == 0:
            self.identification_No = 0
            self.lon_theta0 = np.array([[0],[0],[0]])
            self.lon_beta = np.array([[0],[0],[0]])
            return 0,0,0
        
        if self.identification_No != self.first_lead_vehicle["No"]:
            self.identification_No = 0
            self.lon_theta0 = np.array([[0],[0],[0]])
            self.lon_beta = np.array([[0],[0],[0]])
        
         # 进入纵向风格辨识
        lon_f_xdot = np.transpose(np.array([[1,self.timestep,-self.timestep],
                                            [0,1,0],
                                            [0,0,1]])) #3X3
        lon_f_udot = np.array([[-0.5*self.timestep**2,0,self.timestep]]) #1X3
        lon_l_xdot = np.array([[2*(self.first_lead_vehicle["X_distance"] - self.x_des),0,0],
                               [0,0,0],
                               [0,2*(self.first_lead_vehicle["lon_speed"]-self.v_des),0]]) #3X3
        lon_l_udot = np.array([[0,0,2*self.first_lead_vehicle["lon_a"]]])  #1X3
        lon_F = np.hstack((lon_l_udot,lon_f_udot)) #1X6
        lon_G1 = np.hstack((np.eye(3),np.zeros((3,3))))
        lon_G2_1 = np.dot(-np.linalg.inv(lon_f_xdot),lon_l_xdot) #3X3
        lon_G2_2 = np.linalg.inv(lon_f_xdot) #3X3
        lon_G2 = np.hstack((lon_G2_1,lon_G2_2))  #3X6
        lon_G = np.dot(np.vstack((lon_G1,lon_G2)),self.lon_G0) #6X6
        lon_Q = np.dot(np.transpose(np.dot(lon_F,lon_G)) , np.dot(lon_F,lon_G)) + self.lon_Q0  #6X6
        lon_Q_main = lon_Q[1:,1:] #5X5
        lon_q = lon_Q[1:,0] #5X1
        lon_alpha1 = -np.dot(np.linalg.pinv(lon_Q_main),lon_q.reshape(-1,1)) #5X1
        lon_alpha = np.vstack((np.array([[1]]),lon_alpha1)) #6X1
        lon_theta = np.dot(lon_G1,lon_alpha) #3X1
        if np.linalg.norm(lon_theta-self.lon_theta0,2) < self.lon_error:
            self.lon_theta0 = np.array([[0],[0],[0]])
            self.lon_beta = lon_theta
            self.lon_G0 = np.eye(6)
            self.lon_Q0 = np.zeros((6,6))
        else:
            self.lon_theta0 = lon_theta
            self.lon_G0 = lon_G
            self.lon_Q0 = lon_Q
        
        self.identification_No = self.first_lead_vehicle["No"]

        # return self.identification_No,self.lon_theta0,self.lon_beta

    def GMPC(self, compete_target, data, ego_vehicle, y_des):
        #ACC
        flag, u, f = self.ACC(ego_vehicle)
        if flag == 0:
            pass
        else:
            # print("开启ACC:",u,f)
            return u , f
        #与博弈前车的换道空间决策
        ev_pv_dis = float(data[7].loc[0])
        ev_cv_dis = -float(data[5].loc[0])
        if ev_pv_dis + ev_cv_dis > 4 + ego_vehicle.speed*self.headway + 12:
            if ev_pv_dis >= 4 + ego_vehicle.speed*self.headway and ev_cv_dis >= 12:
                # 是否需要博弈
                if compete_target == False:
                    u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 40, -15, self.v_des, y_des, 
                                        self.timestep, 10, 1, 1, 10, 10, 100)
                    # print("决策无博弈换道:",u,f)
                else:
                    if self.first_lead_vehicle["X_distance"] > self.x_des: x_des = self.first_lead_vehicle["X_distance"]
                    else: x_des = self.x_des
                    try:
                        u,f = utils_v2.GMPC_planner(1, self.timestep, 11, float(self.first_lead_vehicle["X_distance"]),
                                                    ego_vehicle.speed, float(self.first_lead_vehicle["lon_speed"]), ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                                    x_des, self.v_des, self.v_des, y_des, 0, 0,
                                                    self.lon_beta[0,0], self.lon_beta[1,0], self.lon_beta[2,0])
                    except np.linalg.LinAlgError:
                        u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 40, -15, self.v_des, y_des, 
                                        self.timestep, 10, 1, 1, 10, 10, 100)
                    print("决策博弈换道:",u,f)
            elif ev_pv_dis >= 4 + ego_vehicle.speed*self.headway and ev_cv_dis < 12:
                u,f  = self.OMPC(3,ego_vehicle, ego_vehicle.y - ego_vehicle.lateral_pos)
                # print("加速等待换道:",u,f)
            elif ev_pv_dis < 4 + ego_vehicle.speed*self.headway and ev_cv_dis >= 12:
                u,f  = self.OMPC(1,ego_vehicle, ego_vehicle.y - ego_vehicle.lateral_pos)
                # print("减速等待换道:",u,f)
            else:
                u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x, -30, ego_vehicle.speed -10, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        1, 1, 100, 10, 100)
                # print("大力减速等待换道:",u,f)
        else:
            u,f = utils_v2.OMPC_planner(ego_vehicle.x, ego_vehicle.speed, ego_vehicle.y, ego_vehicle.heading - ego_vehicle.lane_heading, 
                                        ego_vehicle.x + 50, -25, ego_vehicle.speed + 10, ego_vehicle.y - ego_vehicle.lateral_pos, self.timestep, 10,
                                        10, 1, 10, 10, 100)
            # print("大力加速等待换道:",u,f)
        
        if u > 3: u = 3
        elif u < -5:u = -5
        else: u = u
                    
        if f > 0.7: f = 0.7
        elif f < -0.7:f = -0.7
        else: f = f

        return u,f

    def collision_detection_old(self,ego_vehicle,time):
        '''碰撞检测'''
        # 碰撞标志
        collision_flag = False
        # 前车 后车 左前车 左后车 右前车 右后车 gap
        x_list = [ego_vehicle.FV[1], ego_vehicle.RV[1], ego_vehicle.LFV[1], ego_vehicle.LRV[1],
                  ego_vehicle.RFV[1],ego_vehicle.RRV[1]]
        y_list = [ego_vehicle.FV[3], ego_vehicle.RV[3], ego_vehicle.LFV[3], ego_vehicle.LRV[3],
                  ego_vehicle.RFV[3],ego_vehicle.RRV[3]]
        for num in range(0,len(x_list)):
            if x_list[num] == -1:
                # 不存在车辆
                continue
            # 先判断纵向间距
            if x_list[num] > 5:
                pass
            else:
                # 纵向间距小于车长 判断横向间距
                 if abs(y_list[num]-ego_vehicle.y) < 2:
                    collision_flag = True
        return collision_flag

