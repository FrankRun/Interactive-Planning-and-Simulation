#!/usr/bin/env python3

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Configure and inspect an instance of CARLA Simulator.

For further details, visit
https://carla.readthedocs.io/en/latest/configuring_the_simulation/
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    print("================ Starting ================")
# 创建一个客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(5)
# 加载OpenDrive地图
    xodr_path = r'/home/com0196/Chery_Tongji/jinghugaosu1.xodr'
    with open(xodr_path, encoding='utf-8') as od_file:
        data = od_file.read()
        vertex_distance = 2.0  # in meters
        max_road_length = 500.0  # in meters
        wall_height = 0.5      # in meters
        extra_width = 1      # in meters
        world = client.generate_opendrive_world(
            data, carla.OpendriveGenerationParameters(
                vertex_distance=vertex_distance,
                max_road_length=max_road_length,
                wall_height=wall_height,
                additional_width=extra_width,
                smooth_junctions=True,
                enable_mesh_visibility=True))
    print("the current world is:", world)
#
except Exception as e:
    print("Exception detected:", e)
finally:
    pass
    print("================ ending ================")
