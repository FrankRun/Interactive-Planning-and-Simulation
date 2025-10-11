# -*- coding: utf-8 -*-
"""
此代码为SUMO仿真环境代码

@author: Frank Yan
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import math
import numpy as np
import ctypes
import sys
import optparse
import random

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib   # noqa
import traci  # noqa

lanewidth =3.75

class Vehicle:
    def __init__(self,ego_id,is_vdm):
        self.step=0
        self.ego_id=ego_id
        self.laneindex = traci.vehicle.getLaneIndex(ego_id)
        self.speed = traci.vehicle.getSpeed(ego_id)
        self.lastspeed=self.speed
        self.initial_speed = self.speed
        self.acceleration=traci.vehicle.getAcceleration(ego_id)
        self.lastacceleration=traci.vehicle.getAcceleration(ego_id)
        self.mingap=traci.vehicle.getMinGap(ego_id)
        self.length = traci.vehicle.getLength(ego_id)
        self.FVid =  traci.vehicle.getLeader(ego_id)[0] if traci.vehicle.getLeader(ego_id) else False
        self.FVdis =  traci.vehicle.getLeader(ego_id)[1]+self.mingap+self.length if self.FVid else 100
        self.FVspeed = traci.vehicle.getSpeed(self.FVid) if self.FVid else self.speed

        self.LFVid = traci.vehicle.getNeighbors(ego_id,0b00000010)[0][0] if traci.vehicle.getNeighbors(ego_id,0b00000010) else False
        self.LFVdis = traci.vehicle.getNeighbors(ego_id,0b00000010)[0][1]+self.mingap +self.length if self.LFVid else 100
        self.LFVspeed = traci.vehicle.getSpeed(self.LFVid) if self.LFVid else self.speed

        self.RFVid = traci.vehicle.getNeighbors(ego_id,0b00000011)[0][0] if traci.vehicle.getNeighbors(ego_id,0b00000011) else False
        self.RFVdis = traci.vehicle.getNeighbors(ego_id,0b00000011)[0][1]+self.mingap+self.length if self.RFVid else 100
        self.RFVspeed = traci.vehicle.getSpeed(self.RFVid) if self.RFVid else self.speed
        '''动力学模型'''
        if is_vdm:
            self.vdm =  ctypes.CDLL('libtoyota.so')
            self.vdm._Z17toyota_initializev()
            self.vdm._Z11toyota_stepdd.restype = ctypes.c_double
            self.vdm._Z11toyota_stepdd.argtypes = [ctypes.c_double,ctypes.c_double]
            for i in range(10):
                self.vdm._Z11toyota_stepdd(0,self.initial_speed)

    def update(self):
        self.step+=1

        self.lastspeed=self.speed
        self.lastacceleration=self.acceleration
        self.speed = traci.vehicle.getSpeed(self.ego_id)
        self.acceleration=traci.vehicle.getAcceleration(self.ego_id)

        self.FVid =  traci.vehicle.getLeader(self.ego_id)[0] if traci.vehicle.getLeader(self.ego_id) else False
        self.FVdis =  traci.vehicle.getLeader(self.ego_id)[1]+self.mingap +self.length if self.FVid else 100
        self.FVspeed = traci.vehicle.getSpeed(self.FVid) if self.FVid else self.speed

        self.LFVid = traci.vehicle.getNeighbors(self.ego_id,0b00000010)[0][0] if traci.vehicle.getNeighbors(self.ego_id,0b00000010) else False
        self.LFVdis = traci.vehicle.getNeighbors(self.ego_id,0b00000010)[0][1]+self.mingap+self.length if self.LFVid else 100
        self.LFVspeed = traci.vehicle.getSpeed(self.LFVid) if self.LFVid else self.speed

        self.RFVid = traci.vehicle.getNeighbors(self.ego_id,0b00000011)[0][0] if traci.vehicle.getNeighbors(self.ego_id,0b00000011) else False
        self.RFVdis = traci.vehicle.getNeighbors(self.ego_id,0b00000011)[0][1]+self.mingap +self.length if self.RFVid else 100
        self.RFVspeed = traci.vehicle.getSpeed(self.RFVid) if self.RFVid else self.speed


class SumoEnv:
    def __init__(self,filedir,config,seed,is_vdm):


        sumoBinary = sumolib.checkBinary('sumo-gui')##'sumo-gui' or 'sumo'
        sumoconfig = filedir+config
        self.startcommand=[sumoBinary, "-c",sumoconfig,'--seed',str(seed)]
        self.ego_id='10000'

        self.statenum=8 ##状态数量 [本车速度，加速度，周围车相对位置，周围车相对速度]
        self.actionnum=1 ##动作数量
        self.max_episode_length = 299
        self.maxacc = 4
        self.timestep = 0.1
        self.warmsteps = 3000
        self.is_vdm=is_vdm
        self.load_network()

    def load_network(self):
        #导入路网
        traci.start(self.startcommand)
        ##预热
        for step in range(self.warmsteps):
            traci.simulationStep()


    def reset(self):
        #生成训练agent
        self.ego_id=str(int(self.ego_id)+1)
        time=traci.simulation.getTime()
        traci.vehicle.add(vehID=self.ego_id,routeID="r0",typeID="inter_AV",depart=time,departLane="random",departSpeed="desired")
        traci.vehicle.highlight(self.ego_id) #将该辆车在路网中标记出来
        traci.gui.trackVehicle("View #0", self.ego_id) #将仿真视角跟随标记的车辆，"View #0"为SUMO默认的视角名称

        id_list = traci.vehicle.getIDList()
        while self.ego_id not in id_list:
            traci.simulationStep()
            id_list = traci.vehicle.getIDList()

        self.egovehicle = Vehicle(self.ego_id,self.is_vdm)
        traci.vehicle.setLaneChangeMode(self.ego_id,0)
        traci.vehicle.setSpeedMode(self.ego_id,0)
        # print(traci.vehicle.getSpeed(self.ego_id))
        self.s_initial=self.get_state()

        return self.s_initial

    def step(self,action):
        '''每一步计算总的reward'''
        acc=action[0]*self.maxacc-2 #-6~2
        self.setaction(acc) ##获取动作
        traci.simulationStep() ##仿真跑一步
        self.egovehicle.update()##本车信息更新
        s_next = self.get_state() ##获取仿真跑一步后的状态
        reward,done = self.reward_compute(s_next) ##计算当前状态-动作的奖励回报
        if done:
            ##remove车辆 清除动力学模型
            traci.vehicle.remove(self.ego_id)
            if self.is_vdm:
                self.egovehicle.vdm._Z16toyota_terminatev()
                libc = ctypes.CDLL(None)  # 加载C标准库
                libc.dlclose(self.egovehicle.vdm._handle)  # 关闭动态库
            traci.simulationStep()

        return s_next,reward,done


    def test(self,action):
        '''每一步计算总的reward'''
        s_current = self.get_state()
        acc=action[0]*self.maxacc-2 #-6~2
        self.setaction(acc) ##获取动作

        traci.simulationStep() ##仿真跑一步

        if self.ego_id not in traci.vehicle.getIDList():
            return s_current,0,True

        self.egovehicle.update()##本车信息更新
        s_next = self.get_state() ##获取仿真跑一步后的状态
        self.max_episode_length = 36000
        reward,done = self.reward_compute(s_next) ##计算当前状态-动作的奖励回报
        if done:
            ##remove车辆
            traci.vehicle.remove(self.ego_id)
            traci.simulationStep()

        return s_next,reward,done


    def get_state(self):
        state=[self.egovehicle.speed,self.egovehicle.acceleration,
               self.egovehicle.FVdis,self.egovehicle.FVspeed-self.egovehicle.speed,
               self.egovehicle.LFVdis,self.egovehicle.LFVspeed-self.egovehicle.speed,
               self.egovehicle.RFVdis,self.egovehicle.RFVspeed-self.egovehicle.speed]

        return state

    def setaction(self,desiredacc):
        if self.is_vdm:
            desiredacc_vdm=self.egovehicle.vdm._Z11toyota_stepdd(desiredacc,self.egovehicle.initial_speed)
            desired_speed=self.egovehicle.speed+desiredacc_vdm*self.timestep
            # print(round(desiredacc,2),'  ',round(desiredacc_vdm,2))
            traci.vehicle.setSpeed(self.ego_id,desired_speed)
        else:
            desired_speed=self.egovehicle.speed+desiredacc*self.timestep
            traci.vehicle.setSpeed(self.ego_id,desired_speed)


    def close(self):
        traci.close()
        sys.stdout.flush()


    def reward_compute(self,state_next):
        '''reward设计
        [速度 纵向acc 前车距离  邻车道前车距离 碰撞]
        机动性：速度
        舒适性：纵向acc
        安全性：距离 碰撞
        '''
        done = False
        theta = [1.5,-0.10,-1,-0.30,-50]##,-0.2]

        # 速度
        max_speed = 100/3.6##130/3.6
        ego_longitudial_speed = state_next[0]/max_speed
        #超速惩罚
        if ego_longitudial_speed>1:
            ego_longitudial_speed=2*(1-ego_longitudial_speed)

        ##舒适性
        ego_longitudial_acc = abs(state_next[1])

        if ego_longitudial_acc <= 1:
            ego_longitudial_acc =0

        ##前车距离 time gap front (THWF)
        FV_rel_distance = state_next[2]
        if FV_rel_distance > 20:
            ego_THWF = 0
        else:
            if state_next[0] <1:
                ego_THWF = 20
            else:
                ego_THWF = state_next[2]/state_next[0]
            ego_THWF = math.exp(-ego_THWF)

        ##左前车距离 time gap rear (THWR)
        LFV_rel_distance = state_next[4]
        if LFV_rel_distance>20:
            ego_THWLFV = 0
        else:
            if state_next[0] <1:
                ego_THWLFV = 20
            else:
                ego_THWLFV = state_next[4]/state_next[0]
            ego_THWLFV= math.exp(-ego_THWLFV)

        ##右前车距离 time gap rear (THWR)
        RFV_rel_distance = state_next[6]
        if RFV_rel_distance>20:
            ego_THWRFV = 0
        else:
            if state_next[0] <1:
                ego_THWRFV = 20
            else:
                ego_THWRFV = state_next[6]/state_next[0]
            ego_THWRFV= math.exp(-ego_THWRFV)

        ##collision
        collision = 0

        if state_next[2] < 15:

            if self.ego_id in traci.simulation.getCollidingVehiclesIDList():
                collision =1
                print('collosion type1')

            if state_next[2]+state_next[3]*self.timestep<traci.vehicle.getLength(self.egovehicle.FVid)+self.egovehicle.mingap+0.5:
                collision =1
                print('collosion type2')

        if collision == 1 or self.egovehicle.step>=self.max_episode_length:
            done = True

        features = np.array([ego_longitudial_speed,
                             ego_longitudial_acc,
                             ego_THWF,ego_THWLFV+ego_THWRFV,
                             collision])


        reward = np.dot(features,theta)

        if  state_next[0]<-1000 or self.ego_id not in traci.vehicle.getIDList(): ##本车已经消失
            reward=-30
            done=True

        return reward, done


