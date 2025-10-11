#!/bin/bash

# 设置CARLA的路径
CARLA_PATH="/home/com0196/wangjie/CARLA"

# 设置加载路网的Python脚本
PYTHON_SCRIPT1="/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/load_xodr.py"
PYTHON_SCRIPT2="/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/run_synchronization_info2_camera.py"

# 设置加载路网
ROAD_SCRIPT="/home/com0196/wangjie/CARLA/Co-Simulation/Sumo/examples/jinghugaosu.sumocfg"

# 启动CARLA服务器
echo "Starting CARLA server..."
${CARLA_PATH}/CarlaUE4.sh &

# 等待CARLA服务器启动
echo "Waiting for CARLA server to start..."
sleep 10  # 根据你的实际情况调整等待时间

# 运行加载路网的Python脚本
echo "Loading road network..."
python3 ${PYTHON_SCRIPT1}

echo "Road network loaded successfully."
sleep 10
