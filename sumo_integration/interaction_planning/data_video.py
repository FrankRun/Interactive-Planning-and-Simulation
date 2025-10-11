import matplotlib.pyplot as plt
import imageio
import pandas as pd
from matplotlib.patches import Rectangle
import numpy as np
#plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def tra_plot(csv_file,scenario,num,save_path):
    '''画场景中本车及周围车辆的轨迹图'''
    data = pd.read_csv(csv_file)
    groups = data.groupby('Time')
    timestep = 0
    images = []
    for No, group in groups:
        timestep += 1
        group = group.reset_index(drop = True)
        is_first=False
        fig, ax = plt.subplots(1,1,figsize=(21,4))
        plt.subplots_adjust(bottom=0.20,top=0.80,left=0.05, right=0.95)
        line = np.linspace(-50, 50,5)
        plt.plot(line, [0]*len(line), '--', color = 'gray', lw = 1)
    
        for i in range(len(group)):
            vehdata = group.iloc[i]
            x=vehdata['X']
            y=vehdata['Y']
            if vehdata['Type'] == 'ego':
                plt.plot(x, y,color='red',linewidth=2, zorder=3)
                rect = Rectangle((x - 2, y - 1), 4, 2, edgecolor='red',facecolor='red', zorder=3,alpha=0.6,linewidth=2,label='Ego vehicle')
                ax.add_patch(rect)
            else:
                plt.plot(x, y,color="black",linewidth=2)
                rect = Rectangle((x - 2, y - 1), 4, 2, edgecolor='black',facecolor='black', zorder=3,alpha=0.6,linewidth=2,label='Surrounding vehicle')
                ax.add_patch(rect)
                         
        plt.ylim(-3.5,3.5)
        plt.xlim(-50,50)
        #plt.legend(fontsize=14)
        plt.ylabel('Y(m)',fontsize=14)
        plt.xlabel('X(m)',fontsize=14)
        plt.title(scenario+f"_number{timestep}",fontsize=14)
        plt.savefig(save_path + f"simulation{num}_tra{timestep}.png", dpi = 300, format = 'png')
        plt.close()
        images.append(imageio.imread(save_path + f"simulation{num}_tra{timestep}.png"))
    
    imageio.mimsave(save_path + "animation.gif", images, fps=5)  # 将图片保存为GIF，fps设置帧率

if __name__ == '__main__':
    tra_plot('/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_3/simulation_0/trajectory.csv', 'OnRamp', 0, 
             '/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_3/simulation_0/figure/')