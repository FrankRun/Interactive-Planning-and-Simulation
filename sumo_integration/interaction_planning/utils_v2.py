# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:50:52 2024

@author: 15834
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize 
import threading
import ctypes
import shutil
import csv
ll = ctypes.cdll.LoadLibrary

'''选择主车 换道场景'''
def egoSelect(vissim_com):
    all_veh_attributes = vissim_com.Net.Vehicles.GetMultipleAttributes(('No', 'Pos', 'Lane')) 
    all_veh_attributes = [veh_attr for veh_attr in all_veh_attributes if veh_attr[2]=="1-1"]
    all_veh_attributes = sorted(all_veh_attributes,key = lambda x: float(x[1]))
    if len(all_veh_attributes) > 0:
        egoNo = int(all_veh_attributes[0][0])
        vissim_com.Net.Vehicles.ItemByKey(egoNo).SetAttValue('VehType', "630")
        return egoNo
    else:
        return 0
def if_cutin(vissim_com, egoNo):
    all_veh_attributes = vissim_com.Net.Vehicles.GetMultipleAttributes(('No', 'Pos', 'Lane')) 
    all_veh_attributes = sorted(all_veh_attributes,key = lambda x: float(x[1]))
    vehicles_lane2 = [veh_attr for veh_attr in all_veh_attributes if veh_attr[2]=="1-2" or int(veh_attr[0])==egoNo]
    vehiclesNo_lane2 = [int(veh_attr[0]) for veh_attr in vehicles_lane2]
    ego_index = vehiclesNo_lane2.index(egoNo)
    if ego_index == 0:
        ## 主车后边没车，不算加塞
        return 0
    else:
        fv_index = ego_index - 1
        if float(vehicles_lane2[ego_index][1]) - float(vehicles_lane2[fv_index][1]) > 75:
            return 0
        else:
            return 1            
    
def delete_files(folder_path):
    shutil.rmtree(folder_path)
    print(f"成功删除文件夹: {folder_path}")

def data_record(vissim_com, log_path, simulation_time) :
    all_veh_attributes = vissim_com.Net.Vehicles.GetMultipleAttributes(('No', 'VehType', 'Pos','Speed','Acceleration',
                                                                      'CoordFrontX','CoordFrontY','CoordRearX','CoordRearY','Lane','acc_com_uda', "angle_com_uda"))
    with open(log_path + 'trajectory.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for veh_attr in all_veh_attributes:
            data = [simulation_time, int(veh_attr[0]), int(veh_attr[1]), int(veh_attr[9][2]), 0.5*(float(veh_attr[5])+float(veh_attr[7])),
                    0.5*(float(veh_attr[6])+float(veh_attr[8])), float(veh_attr[3]), round(float(veh_attr[10]),2), round(float(veh_attr[11]),2)]
            writer.writerow(data)



def cum_prod(S, x_0):
    N = S.shape[2]
    m = x_0.shape[0]
    P = np.eye(m)
    for i in range(N):
        P = np.dot(P, S[:, :, i])
    return P


    
def OMPC_planner(x0,v0,y0,fi0,x1, x_des,v1, y_des, step,N,w1,w2,w3, w4,w5):
    '''
    x0:初始位置   v0:初始速度  y0:初始纵向距离 fi0:初始航向角 x1 前车位置 v_des:期望速度 y_des: 期望车道  v1:前车速度    step:步长   N:预测时域
    w1,w2,w3,w4:分别是速度、加速度、纵向距离、航向角权重
    '''
    #定义初始状态和时间步长
    x_0 = np.array([x0, v0, y0, fi0]).reshape(-1, 1)
    X_des=np.array([x1+x_des, v1, y_des, 0],dtype=object).reshape(-1, 1)
    #定义系统动态方程 x(k+1) = A*x(k) + B*u(k)
    A = np.array([[1, step, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, -v0*step],
                  [0, 0, 0, 1]])        
    B = np.array([[0.5 * step ** 2, 0],
                  [step, 0],
                  [0, 0],
                  [0, step * v0/4]])
    #定义目标状态和控制输入的权重矩阵 Q 和 R
    Q = np.diag([w1, w2, w4, w5])
    if v0*3.6 > 80:
        w6 = 20000
    elif v0*3.6 >60 and v0*3.6<=80:
        w6 = 15000
    elif v0*3.6>=45 and v0*3.6 <= 60:
        w6 = 5000
    elif v0*3.6>=30:
        w6 = 2500
    else:
        w6 = 1000
    R =np.diag([w3, w6])
    #转化为用控制量ut表示的，关于状态量的推导方程的矩阵
    M=np.eye(4)
    C=np.zeros((4,2*N))
    C = np.concatenate((C, np.kron(np.diag(np.ones(N)), B)), axis=0)
    Qt = np.kron(np.diag(np.concatenate((np.ones(N), np.array([0])))), Q)
    Rt = np.kron(np.diag(np.ones(N)), R)
    
    for i in range(1, N+1):
        M = np.vstack((M, np.linalg.matrix_power(A, i)))
        if i == 1:
            pass  # 与Matlab中的 '1;' 相对应，这里什么都不做
        else:
            for j in range(1, i):
                C[4*i:4*i+4, 2*j-2:2*j] = np.dot(np.linalg.matrix_power(A, i-j), B)
    # 更新X_des
        X_des = np.vstack((X_des, np.array([x1 + i*step*v1 + x_des, v1, y_des, 0],dtype=object).reshape(-1, 1)))
    
    u_max = np.array([3, 0.7]).reshape(-1,1)
    u_min = np.array([-5, -0.7]).reshape(-1,1)
    U_max = np.tile(u_max, (N, 1))
    U_min = np.tile(u_min, (N, 1))
    H = C.T @ Qt @ C + Rt
    f = 2 * (x_0.T @ M.T - X_des.T) @ Qt @ C
    obj_func = lambda x: 0.5 * x.T @ H @ x + f @ x
    bounds = [(low, high) for low, high in zip(U_min.flatten(), U_max.flatten())]
    x0 = np.zeros(20)
    result = minimize(obj_func, x0, method='SLSQP', bounds=bounds, options={'maxiter': 100000})
    sol = result.x
    acc = sol[0]
    angle = sol[1]
    return acc,angle

def GMPC_planner(Ti, step, N, x0, v10, v20, y0, w0, x_des, v1_des, v2_des, y1_des, y2_des, w_des, beta1, beta2, beta3):
    x_0 = np.array([x0, v10, v20, y0, w0]).reshape(-1, 1)
    x2_desired = np.hstack((x_0, np.tile(np.array([x_des, v1_des, v2_des, y2_des, w_des]).reshape(-1, 1), (1, 10))))
    x1_desired = np.hstack((x_0, np.tile(np.array([x_des, v1_des, v2_des, y1_des, w_des]).reshape(-1, 1), (1, 10))))
    #权重矩阵计算
    Q_2 = np.diag([beta1, 0, beta2, 0, 0])
    Q_N_2 = np.diag([beta1, 0, beta2, 0, 0])
    Q_1 = np.diag([1, 1, 0, 5, 10])
    Q_N_1 = np.diag([10, 10, 0, 10, 50])
    R_2 = np.diag([beta3, 1])
    R_1 = np.diag([25, 1000])
    S_1 = np.zeros((5, 2))
    #维数
    u = Q_2.shape[0]
    v = R_2.shape[1]
    ll = v * (N - 1)
    mm = u * N
    nn = u * (N - 1)
    O = np.zeros((v, 1, N))
    x = np.zeros((5, N))
    # 变量初始值
    v1 = v10 * np.ones((1, N))
    u_new = np.zeros((v, N - 1))
    A = np.zeros((u, u, N))
    B = np.zeros((u, v, N))
    C = np.zeros((u, v, N))
    EE = np.zeros((ll, ll))
    T = np.zeros((u, v, N))
    FF = np.zeros((ll, 1))
    S = np.zeros((u, u, N))
    # 创建具有随机元素的三维NumPy数组
    J = np.array([np.random.rand(2, 5) for _ in range(11)])
    G = np.array([np.random.rand(2, 2) for _ in range(11)])
    H = np.array([np.random.rand(2, 5) for _ in range(11)])
    A_ = np.array([np.random.rand(5, 5) for _ in range(10)])
    B_ = np.array([np.random.rand(5, 2) for _ in range(10)])
    C_ = np.array([np.random.rand(5, 2) for _ in range(10)])
    Q_ = np.array([np.random.rand(5, 5) for _ in range(11)])
    R_ = np.array([np.random.rand(2, 2) for _ in range(10)])
    #迭代
    for i in range(N):
        A[:, :, i] = np.array([[1, step, -step, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 1, -v1[0][i] * step],
                      [0, 0, 0, 0, 1]])        
        B[:, :, i] = [[0.5 * step ** 2, 0],
                      [step, 0],
                      [0, 0],
                      [0, 0],
                      [0, step * v1[0][i] / 4]]
    
        C[:, :, i] = [[-0.5 * step ** 2, 0],
                      [0, 0],
                      [step, 0],
                      [0, 0],
                      [0, 0]]
    P = np.zeros((5, 5, N))
    P[:, :, N-1] = Q_N_2
    O[:, :, N-1] = np.zeros((v, 1))
    x[:, 0] = np.array([x0, v10, v20, y0, w0])
    for k in range(N-2, -1, -1):
        P[:, :, k] = A[:, :, k].T @ P[:, :, k+1] @ A[:, :, k] + Q_2 - A[:, :, k].T @ P[:, :, k+1] @ C[:, :, k] @ np.linalg.inv(C[:, :, k].T @ P[:, :, k+1] @ C[:, :, k] + R_2) @ C[:, :, k].T @ P[:, :, k+1] @ A[:, :, k] - S_1 @ np.linalg.inv(C[:, :, k].T @ P[:, :, k+1] @ C[:, :, k] + R_2) @ (S_1.T + 2 * C[:, :, k].T @ Q_2 @ A[:, :, k])
    # 选取第一个时间步的矩阵
    B_[0] = B[:, :, 0]
    A_[0] = A[:, :, 0]
    C_[0] = C[:, :, 0]
    Q_[0] = Q_1
    R_[0] = R_1
    
    for j in range(1, N):
      H[j] = -np.linalg.inv(C[:, :, j-1].T @ P[:, :, j] @ C[:, :, j-1] + R_2) @ C[:, :, j-1].T  
      J[j] = H[j] @ P[:, :, j] @ A[:, :, j-1]
      G[j] = H[j] @ P[:, :, j] @ B[:, :, j-1]
      FF[2*j-2:2*j, 0] = -H[j] @ P[:, :, j] @ x2_desired[:, j]
      
      S[:, :, j] = A[:, :, j-1].T @ (np.eye(u) - C[:, :, j-1] @ np.linalg.inv(C[:, :, j-1].T @ P[:, :, j] @ C[:, :, j-1] + R_2) @ C[:, :, j-1].T @ P[:, :, j]).T
      T[:, :, j] = S[:, :, j] @ P[:, :, j] @ B[:, :, j-1]
      
      if j != N-1:
        B_[j] = B[:, :, j]
        C_[j] = C[:, :, j]
        A_[j] = A[:, :, j]
        Q_[j] = Q_1
        R_[j] = R_1
      else:
        Q_[j] = Q_N_1
    JJ = np.hstack([block_diag(*J[1:,:,:]), np.zeros((ll, u))])
    GG = block_diag(*G[1:,:,:])
    # 对于AA, BB, CC的构建，首先构建上半部分的零矩阵，然后构建下半部分
    AA_upper = np.zeros((u, mm))
    AA_lower = np.hstack([block_diag(*A_[0:]), np.zeros((nn, u))])
    AA = np.vstack([AA_upper, AA_lower])
    
    BB_upper = np.zeros((u, ll))
    BB_lower = block_diag(*B_[0:])
    BB = np.vstack([BB_upper, BB_lower])
    CC_upper = np.zeros((u, ll))
    CC_lower = block_diag(*C_[0:])
    CC = np.vstack([CC_upper, CC_lower])
    DD = np.vstack([x_0.reshape(-1, 1), np.zeros((nn, 1))])  # 确保x_0是列向量
    
    # QQ和RR使用block_diag构建，然后用它们构建PP
    QQ = block_diag(*Q_[0:])
    RR = block_diag(*R_[0:])
    PP = block_diag(QQ, RR)
    
    # 对于OO的构建，先构建中间矩阵，然后重塑和拼接
    OO_mid = np.hstack([-Q_1 @ x1_desired[:, 0:x1_desired.shape[1]-1], -Q_N_1 @ x1_desired[:, -1].reshape(-1, 1)])
    OO = np.vstack([OO_mid.flatten(order='F').reshape(-1, 1), np.zeros((ll, 1))])  
    for i in range(1, N-1):  # 注意Python索引从0开始，MATLAB从1开始
        for j in range(1, N-1):
            if i < j:
                EE[2*i-2:2*i, 2*j:2*j+2] = H[i] @ cum_prod(S[:,:,i+1:j+1], x_0) @ T[:, :, j+1]
            elif i == j:
                EE[2*i-2:2*i, 2*j:2*j+2] = np.dot(H[i], T[:, :, j+1])
            else:  # i > j
                EE[2*i-2:2*i, 2*j:2*j+2] = np.zeros((v, v))
    I = np.eye(mm)
    W = np.hstack([I - AA - np.dot(CC, JJ), -np.dot(CC, GG+EE) - BB])
    V = np.dot(CC, FF) + DD
    Aeq_1 = np.hstack([PP,W.T])
    Aeq_2 = np.hstack([W,np.zeros((mm,mm))])
    Aeq = np.vstack([Aeq_1,Aeq_2])
    beq = np.vstack([-OO,V])
    sol = np.linalg.solve(Aeq, beq)
    acc = sol[55::2][:10]
    angle = sol[56::2][:10]
    angle[angle>0.7] = 0.7; angle[angle<-0.7] = -0.7
    acc[acc>3] = 3; acc[acc<-5] = -5
    #优化
    # x_max = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]).reshape(-1,1)
    # u_max = np.array([3, 0.7]).reshape(-1,1)
    # x_min = np.array([-np.inf, 0, 0, -np.inf, -np.inf]).reshape(-1,1)
    # u_min = np.array([-3, -0.7]).reshape(-1,1)
    
    # X_max = np.tile(x_max, (N, 1))
    # X_min = np.tile(x_min, (N, 1))
    # U_max = np.tile(u_max, (N-1, 1))
    # U_min = np.tile(u_min, (N-1, 1))
    # ub = np.concatenate((X_max, U_max), axis=0)
    # lb = np.concatenate((X_min, U_min), axis=0)
    # obj_func = lambda x: x @ PP @x.T + OO.T @ x
    # lin_constraint = {'type': 'eq', 'fun': lambda x: np.dot(W, x).flatten() - V.flatten()}
    # bounds = [(low, high) for low, high in zip(lb.flatten(), ub.flatten())]
    # x0 = np.zeros(75)  # 根据实际情况调整
    # # 执行优化
    # result = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=(lin_constraint), options={'maxiter': 100000})
    # sol = result.x
    # acc = sol[55::2]
    # angle = sol[56::2]
    u_ev = sol[55:][:20].reshape(-1,1)
    x = sol[:55].reshape(-1,1)
    u_cv = np.dot(JJ, x) + np.dot(GG+EE, u_ev) + FF
    u_cv = u_cv[0::2]
    return acc[0][0],angle[0][0]


def cv_trajectory(competeVehicle, u_cv, step):
    '''
    Parameters
    ----------
    competeVehicle : vissim_object
        vissim vehicle object.
    u_cv : np.array
        fault 10*1 type of np.array.

    Returns
    -------
    None.

    '''
    x_position = (float(competeVehicle.AttValue('CoordFrontX'))+float(competeVehicle.AttValue('CoordRearX')))/2
    velocity = [float(competeVehicle.AttValue("Speed")/3.6)]
    cv_trajectory = [x_position]
    for t in range(u_cv.shape[0]):
        velocity.append(velocity[t] +  u_cv[t][0]*step)
        cv_trajectory.append(cv_trajectory[t] + velocity[t] * step + 0.5 * u_cv[t][0]*step**2)
    return cv_trajectory
        
    

#%%    
# if __name__ == '__main__':
#      acc,angle,u_ev,u_cv = GMPC_planner(1, 0.1, 11, 10, 15, 15, 3.5, 0, 15, 20, 20, 0, 0, 0, 1, 1, 1)
#%%
def GMPC_planner_v2(Ti, step, N, x0, v10, v20, y0, w0, x_des, v1_des, v2_des, y1_des, y2_des, w_des, beta1, beta2, beta3):
    
    lib_acc = ll(r"GMPC_dll\acc_control.dll")
    lib_acc.acc_control.restype = ctypes.c_double
    lib_steer = ll(r"GMPC_dll\steer_control.dll")
    lib_steer.steer_control.restype = ctypes.c_double
    lib_reaction = ll(r"GMPC_dll\cv_reaction.dll")
    lib_reaction.cv_reaction.restype = ctypes.POINTER(ctypes.c_double)
    acc = lib_acc.acc_control(ctypes.c_double(Ti),  #Ti
                                ctypes.c_double(step), #step
                                ctypes.c_double(N),  #N
                                ctypes.c_double(x0), #delta_x
                                ctypes.c_double(v10), #v_ev
                                ctypes.c_double(v20), #v_cv
                                ctypes.c_double(y0), #y_ev
                                ctypes.c_double(w0), #efi
                                ctypes.c_double(x_des), #防霸凌期望间距
                                ctypes.c_double(v1_des), #对手车期望间距
                                ctypes.c_double(v2_des), #防霸凌期望速度
                                ctypes.c_double(y1_des), #对手车期望速度
                                ctypes.c_double(y2_des), #对手车期望车道
                                ctypes.c_double(w_des), #对手车期望航向角
                                ctypes.c_double(beta1), #纵向辨识参数
                                ctypes.c_double(beta2), #纵向辨识参数
                                ctypes.c_double(beta3) #纵向辨识参数
                                )
    angle = lib_steer.steer_control(ctypes.c_double(Ti),  #Ti
                                ctypes.c_double(step), #step
                                ctypes.c_double(N),  #N
                                ctypes.c_double(x0), #delta_x
                                ctypes.c_double(v10), #v_ev
                                ctypes.c_double(v20), #v_cv
                                ctypes.c_double(y0), #y_ev
                                ctypes.c_double(w0), #efi
                                ctypes.c_double(x_des), #防霸凌期望间距
                                ctypes.c_double(v1_des), #对手车期望间距
                                ctypes.c_double(v2_des), #防霸凌期望速度
                                ctypes.c_double(y1_des), #对手车期望速度
                                ctypes.c_double(y2_des), #对手车期望车道
                                ctypes.c_double(w_des), #对手车期望航向角
                                ctypes.c_double(beta1), #纵向辨识参数
                                ctypes.c_double(beta2), #纵向辨识参数
                                ctypes.c_double(beta3) #纵向辨识参数
                                )
    return acc,angle