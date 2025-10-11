import numpy as np
import pandas as pd
import warnings
import os
import math
import csv
from sumo_integration.interaction_planning.antibully_model import AntiBully
from sumo_integration.interaction_planning.antibully_model_v2 import Antibully_ACC
# 设置显示所有行
pd.set_option('display.max_rows', None)
# 设置显示所有列
pd.set_option('display.max_columns', None)

class interaction_planning:
    '''
    交互式规控功能实现的核心class 考虑在carla_simulation初始化时实例化
    '''
    def __init__(self,world, scenario, num, success_rate):
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        '''CARLA API'''
        self.world = world
        '''读入决策数据集'''
        self.dataset = np.load(r"/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/sumo_integration/interaction_planning/dataset.npy")
        self.dataset = pd.DataFrame(self.dataset)
        '''读入必要的初始化参数'''
        self.timestep = 0.1
        self.step = 0
        self.decision = 3
        self.Exit = False
        self.if_decision = False
        self.if_in = False
        self.initial_y = 1000
        self.initial_time = False
        self.ego_id = '10000'
        self.planner = AntiBully(self.world)
        self.planner_acc = Antibully_ACC(self.world)
        '''控制量'''
        self.if_start_refine = False
        self.old_u = 0
        '''log'''
        # log
        self.log_simulation_path = f"/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_{scenario}/simulation_{success_rate}/"
        if not os.path.exists(self.log_simulation_path ):
            os.mkdir(self.log_simulation_path)
        with open(self.log_simulation_path + 'trajectory.csv', mode='w', newline='') as file:
            header =  ['Time', 'No', 'Type', 'X','Y','Speed','Acc_com', 'Angle_com']
            writer = csv.writer(file)
            writer.writerow(header)

    def data_record(self,ego_id, ego_vehicle, time):
        with open(self.log_simulation_path + 'trajectory.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if ego_vehicle.FV[0] != False:
                data = [time, ego_vehicle.FV[0], "FV", ego_vehicle.FV[1], ego_vehicle.FV[3] - ego_vehicle.y, ego_vehicle.FV[2], ego_vehicle.FV[4],0]
                writer.writerow(data)
            if ego_vehicle.LFV[0] != False:
                data = [time, ego_vehicle.LFV[0], "LFV", ego_vehicle.LFV[1], ego_vehicle.LFV[3] - ego_vehicle.y, ego_vehicle.LFV[2], ego_vehicle.LFV[4],0]
                writer.writerow(data)
            if ego_vehicle.LFFV[0] != False:
                data = [time, ego_vehicle.LFFV[0], "LFFV", ego_vehicle.LFFV[1], ego_vehicle.LFFV[3] - ego_vehicle.y, ego_vehicle.LFFV[2], ego_vehicle.LFFV[4],0]
                writer.writerow(data)
            if ego_vehicle.LRV[0] != False:
                data = [time, ego_vehicle.LRV[0], "LRV", -ego_vehicle.LRV[1], ego_vehicle.LRV[3] - ego_vehicle.y, ego_vehicle.LRV[2], ego_vehicle.LRV[4],0]
                writer.writerow(data)
            if ego_vehicle.LRRV[0] != False:
                data = [time, ego_vehicle.LRRV[0], "LRRV", -ego_vehicle.LRRV[1], ego_vehicle.LRRV[3] - ego_vehicle.y, ego_vehicle.LRRV[2], ego_vehicle.LRRV[4],0]
                writer.writerow(data)
            
            data = [time, ego_id, "ego", 0, 0, ego_vehicle.speed, ego_vehicle.acceleration,ego_vehicle.heading - ego_vehicle.lane_heading]
            writer.writerow(data)

    def control_refine(self,ego_vehicle,u,f):
        if not self.if_start_refine:
            self.old_u = u
            self.if_start_refine = True
            return u,f
        # acceleration refine
        if u - self.old_u >= 2: u = self.old_u + 2
        elif u - self.old_u <= -2: u = self.old_u - 2
        else: u = u
        # heading refine
        heading_change = ego_vehicle.speed*0.1*f/4
        if heading_change >= 0.06: f = 4*0.06*10/ego_vehicle.speed
        elif heading_change <= -0.06: f =  -4*0.06*10/ego_vehicle.speed
        else: f = f

        self.old_u = u
        return u,f
    
    def exit(self,ego_vehicle):
        if abs(abs(ego_vehicle.y - self.initial_y) - 3.88) < 0.1 and abs(ego_vehicle.heading - ego_vehicle.lane_heading) < 0.05:
            return True
        else:
            return False
    
    def exit_acc(self,time):
        if time - self.initial_time > 10:
            return True
        else:
            return False


    def run_simulation(self,ego_id, ego_vehicle, time):
        '''输入主车编号 判断主车是否进入环境'''
        compete_target=False
        if ego_id == 0:
            return 0
        '''导入SUMO采集的数据并处理'''
        data = self.planner.data_update_left(ego_vehicle,time)
        #print(ego_vehicle.lateral_pos)
        '''交互式决策'''
        if self.step >= 20 and self.step % 10 == 0:
            if self.decision == 2:
                #print('already decision:',self.decision)
                pass
            else:
                self.decision = self.planner.decision_making(data,self.dataset)
                #print('decision:',self.decision)
        self.step += 1
        '''特定路段补丁
        if ego_vehicle.edge == '-68' and ego_vehicle.x < 50:
            self.decision = 3
            self.if_in = False
        '''
        f = open(self.log_simulation_path + 'debug_decision.txt', 'a')
        print("ego_decision:", self.decision, file = f)
        f.close()
        '''初始横向位置'''
        if self.if_in == False and self.decision == 2:
            self.initial_y = ego_vehicle.y - ego_vehicle.lateral_pos
            self.if_in = True
        '''有限状态机决策'''
        if self.decision == 2:
            compete_target = self.planner.select_competevehicle(ego_vehicle)
            if compete_target != False:
                self.planner.behaviour_identification()
                # 博弈式换道
                u,f = self.planner.GMPC(compete_target,data,ego_vehicle, self.initial_y+3.88)
            else:
                # 不存在博弈车，正常换道
                u,f = self.planner.GMPC(compete_target,data,ego_vehicle, self.initial_y+3.88)
        else:
            # 决策不为换道
            u,f = self.planner.OMPC(self.decision,ego_vehicle, self.initial_y+3.88)
        '''控制微调'''
        u,f = self.control_refine(ego_vehicle,u,f)
        #print("加速度:",u,"方向盘转角:",f)
        '''数据记录'''
        self.data_record(ego_id, ego_vehicle, time)
        '''是否退出'''
        self.Exit = self.exit(ego_vehicle)
        
        delta_s=self.timestep*ego_vehicle.speed+0.5*u*self.timestep**2
        delta_fi=self.timestep*ego_vehicle.speed*math.tan(f)/4
        delta_l=-self.timestep*ego_vehicle.speed*math.sin(ego_vehicle.heading+delta_fi-ego_vehicle.lane_heading)
        delta_fi=delta_fi/math.pi*180
        theta=ego_vehicle.lane_heading

        delta_x=delta_s*math.sin(theta)-delta_l*math.cos(theta)
        delta_y=delta_s*math.cos(theta)+delta_l*math.sin(theta)

        return self.Exit, delta_x, delta_y, delta_fi, u, compete_target
    
    def run_simulation_acc(self,ego_id, ego_vehicle, time):
        '''输入主车编号 判断主车是否进入环境'''
        compete_target=False
        if ego_id == 0:
            return 0
        '''时间记录'''
        if self.initial_time == False:
            self.initial_time = time
        cutin_vehicle, pre_vehicle = self.planner_acc.data_predeal(ego_vehicle)
        identification_No,lon_beta,lat_beta = self.planner_acc.behaviour_identification(ego_vehicle)
        u,f = self.planner_acc.planning(ego_vehicle)
        '''控制微调'''
        u,f = self.control_refine(ego_vehicle,u,f)
        '''数据记录'''
        self.data_record(ego_id, ego_vehicle, time)
        '''是否退出'''
        self.Exit = self.exit_acc(time)

        delta_s=self.timestep*ego_vehicle.speed+0.5*u*self.timestep**2
        delta_fi=self.timestep*ego_vehicle.speed*math.tan(f)/4
        delta_l=-self.timestep*ego_vehicle.speed*math.sin(ego_vehicle.heading+delta_fi-ego_vehicle.lane_heading)
        delta_fi=delta_fi/math.pi*180
        theta=ego_vehicle.lane_heading

        delta_x=delta_s*math.sin(theta)-delta_l*math.cos(theta)
        delta_y=delta_s*math.cos(theta)+delta_l*math.sin(theta)

        return self.Exit, delta_x, delta_y, delta_fi, u, False

        



        
        