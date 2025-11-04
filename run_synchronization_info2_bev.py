#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import argparse
import logging
import time
import random
import shutil
import traci
from RL.ppo import PPO_continuous
# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================


class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(self,
                 sumo_simulation,
                 carla_simulation,
                 tls_manager='none',
                 sync_vehicle_color=False,
                 sync_vehicle_lights=False):

        self.sumo = sumo_simulation
        self.carla = carla_simulation
        self.sumo_ego_id=False
        self.carla_ego_id=False

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights
        
        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        self.control = {
        "delta_x": False,
        "delta_y": False,
        "delta_fi": False,
        "acc":False
        }
        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        traffic_manager = self.carla.client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

    # Initialization
        self.camera = None
        self.camera_spawn_time = None
    def tick(self,flag):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # carla-->sumo sync
        # -----------------
        ##获取本车信息
        ego_vehicle=self.sumo.get_vehicle_info()
        ##本车控制量输出
        collision_flag, exit_flag, delta_x,delta_y,delta_fi,u,compete_target=self.carla.tick(self.carla_ego_id,ego_vehicle)
        self.control["delta_x"]=delta_x
        self.control["delta_y"]=delta_y
        self.control["delta_fi"]=delta_fi
        self.control["acc"]=u

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)

        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(carla_actor.get_transform(),
                                                             carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(sumo_actor.signals,
                                                                     carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

        """Camera 设置"""
        """跟随车辆移动视角"""        
        if flag:
            # 查找Camera蓝图
            camera_bp = self.carla.world.get_blueprint_library().find('sensor.camera.rgb')
            # 设置生成Camera的附加类型为SpringArmGhost
            Atment_SpringArmGhost = carla.AttachmentType.Rigid
            # 设置Camera的安装坐标系
            Camera_transform = carla.Transform(carla.Location(x=-10, y=0, z=60),
                                carla.Rotation(pitch=-60, yaw=0, roll=0))
            if self.carla_ego_id>0:
                actor = self.carla.world.get_actor(self.carla_ego_id)
                # 生成Camera
                self.camera = self.carla.world.spawn_actor(camera_bp, Camera_transform, attach_to=actor,
                                                    attachment_type=Atment_SpringArmGhost)
            else:
                self.camera = False
        # 设置观察者视图
        if self.camera:
            spectator = self.carla.world.get_spectator()
            spectator.set_transform(self.camera.get_transform())
        """固定视角"""
        # Generate camera in carla
        '''
        if flag and self.camera is None:
            # search Camera
            camera_bp = self.carla.world.get_blueprint_library().find('sensor.camera.rgb')
            # Set Camera SpringArmGhost
            Atment_SpringArmGhost = carla.AttachmentType.Rigid
            # Set Camera coordinate
            Camera_transform = carla.Transform(carla.Location(x=-10, y=0, z=10),
                                carla.Rotation(pitch=-10, yaw=0, roll=0))
            # Get actor
            vehicle_list = self.carla.world.get_actors().filter('vehicle.*')
            if vehicle_list:
                actor = random.choice(vehicle_list)
                # Generate Camera
                self.camera = self.carla.world.spawn_actor(camera_bp, Camera_transform, attach_to=actor,
                                                    attachment_type=Atment_SpringArmGhost)
                # 记录生成Camera的时间
                self.camera_spawn_time = time.time()
        
        # 检查是否需要销毁Camera
        if self.camera is not None and (time.time() - self.camera_spawn_time) > 1.0:
            # 获取Camera的位置
            camera_transform = self.camera.get_transform()
            # 销毁Camera
            self.camera.destroy()
            self.camera = None

            # 设置观察者视图
            spectator = self.carla.world.get_spectator()
            spectator.set_transform(camera_transform)
        '''
       # -----------------
        # sumo-->carla sync
        # -----------------
        #sumo运行
        self.sumo.tick(self.control,compete_target)
        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                if sumo_actor_id=='10000':
                    carla_blueprint.set_attribute('color', '255,0,0')
                carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                                   sumo_actor.extent)
                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]
            sumo_actor = self.sumo.get_actor(sumo_actor_id)
            carla_actor = self.carla.get_actor(carla_actor_id)
            if sumo_actor_id=='10000':
                self.carla_ego_id=carla_actor_id

            carla_transform = BridgeHelper.get_carla_transform(sumo_actor.transform,
                                                               sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(carla_actor.get_light_state(),
                                                                   sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)
        # 仿真结束判断
        if collision_flag == True:
            return 'failure'
        if exit_flag == True:
            return 'success'
        else:
            return 'continue'

    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        self.sumo.close()


def synchronization_loop(args,agent,scenario, num, success_rate):
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(agent,args.sumo_cfg_file, args.step_length,num+10, args.sumo_host,
                                     args.sumo_port, args.sumo_gui, args.client_order)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length,scenario, num, success_rate)

    synchronization = SimulationSynchronization(sumo_simulation, carla_simulation, args.tls_manager,
                                                args.sync_vehicle_color, args.sync_vehicle_lights)
    try:
        flag = True
        while True:
            start = time.time()

            exit_flag = synchronization.tick(flag)
            
            #change NPC view
            if flag and synchronization.camera:
                    flag = False

            end = time.time()
            elapsed = end - start
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)
            # print(exit_flag)
            if exit_flag == 'success':
                # synchronization.close()
                return True
            if exit_flag == 'failure':
                # synchronization.close()
                return  False

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    except traci.exceptions.TraCIException:
        return 'skip'
    finally:
        logging.info('Cleaning synchronization')

        synchronization.close()

def delete_files(folder_path):
    shutil.rmtree(folder_path)
    print(f"成功删除文件夹: {folder_path}")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('sumo_cfg_file', type=str, help='sumo configuration file')
    argparser.add_argument('--carla-host',
                           metavar='H',
                           default='127.0.0.1',
                           help='IP of the carla host server (default: 127.0.0.1)')
    argparser.add_argument('--carla-port',
                           metavar='P',
                           default=2000,
                           type=int,
                           help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--sumo-host',
                           metavar='H',
                           default=None,
                           help='IP of the sumo host server (default: 127.0.0.1)')
    argparser.add_argument('--sumo-port',
                           metavar='P',
                           default=None,
                           type=int,
                           help='TCP port to listen to (default: 8813)')
    argparser.add_argument('--sumo-gui', action='store_true', help='run the gui version of sumo')
    argparser.add_argument('--step-length',
                           default=0.10,
                           type=float,
                           help='set fixed delta seconds (default: 0.05s)')
    argparser.add_argument('--client-order',
                           metavar='TRACI_CLIENT_ORDER',
                           default=1,
                           type=int,
                           help='client order number for the co-simulation TraCI connection (default: 1)')
    argparser.add_argument('--sync-vehicle-lights',
                           action='store_true',
                           help='synchronize vehicle lights state (default: False)')
    argparser.add_argument('--sync-vehicle-color',
                           action='store_true',
                           help='synchronize vehicle color (default: False)')
    argparser.add_argument('--sync-vehicle-all',
                           action='store_true',
                           help='synchronize all vehicle properties (default: False)')
    argparser.add_argument('--tls-manager',
                           type=str,
                           choices=['none', 'sumo', 'carla'],
                           help="select traffic light manager (default: none)",
                           default='none')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')

    argparser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    argparser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    argparser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    argparser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    argparser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    argparser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    argparser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    argparser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    argparser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    argparser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    argparser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    argparser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    argparser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    argparser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    argparser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    argparser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    argparser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    argparser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    argparser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    argparser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    argparser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    argparser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    argparser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    argparser.add_argument("--state_dim", type=int, default=8, help="state_dimision")
    argparser.add_argument("--action_dim", type=int, default=1, help="action_dimision")
    
    arguments = argparser.parse_args()

    agent = PPO_continuous(arguments)
    agent.load("RL/PPO_preTrained/chery_highway_60_90_2/",3000000)
    # agent.load("RL/PPO_preTrained/chery_highway_30_60_2/",2500000)
    # agent.load("RL/PPO_preTrained/chery_highway_0_30_4/",3000000)
    # agent.load("RL/PPO_preTrained/chery_highway_aggressive/",3000000)
    # agent.load("RL/PPO_preTrained/chery_highway_conversation/",3000000)
    # agent.load("RL/PPO_preTrained/chery_highway_normal/",3000000)

    if arguments.sync_vehicle_all is True:
        arguments.sync_vehicle_lights = True
        arguments.sync_vehicle_color = True

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    # 测试
    scenario = 16
    log_dir=f'/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_{scenario}/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    success_rate = []
    num = 0
    failed_num = []
    while num < 1000:
        if len(success_rate) == 100:
            print('simulation ending')
            break
        # 仿真运行
        success = synchronization_loop(arguments,agent, scenario, num, len(success_rate))
        # 如果路网问题
        if success == 'skip':
            num += 1
            delete_files(log_dir + f'simulation_{len(success_rate)}')
            continue
        # 路网没问题
        success_rate.append(success)
        f=open(f'/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_{scenario}/A_debug_success_rate.txt',"a")
        print(f"success rate after {len(success_rate)} simuluation:",round(len([k for k in success_rate if k == True])*100/len(success_rate),2),'%',file=f)
        print(f"success rate after {len(success_rate)} simuluation:",round(len([k for k in success_rate if k == True])*100/len(success_rate),2),'%')
        f.close()
        if success == False:
            failed_num.append(num)
            f=open(f'/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/results/scenario_{scenario}/A_debug_failed_scenario.txt',"a")
            print("failed scenario num:",failed_num, file = f)
            f.close()
        num += 1
    
        


