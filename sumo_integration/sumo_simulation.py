#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
""" This module is responsible for the management of the sumo simulation. """

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import collections
import enum
import logging
import os
import ctypes
import carla  # pylint: disable=import-error
import sumolib  # pylint: disable=import-error
import traci  # pylint: disable=import-error
import argparse
from .constants import INVALID_ACTOR_ID
import math
import lxml.etree as ET  # pylint: disable=import-error
from RL.cheryenvironment import Vehicle
from RL.ppo import PPO_continuous
import random
# ==================================================================================================
# -- sumo definitions ------------------------------------------------------------------------------
# ==================================================================================================


# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html#signal_state_definitions
class SumoSignalState(object):
    """
    SumoSignalState contains the different traffic light states.
    """
    RED = 'r'
    YELLOW = 'y'
    GREEN = 'G'
    GREEN_WITHOUT_PRIORITY = 'g'
    GREEN_RIGHT_TURN = 's'
    RED_YELLOW = 'u'
    OFF_BLINKING = 'o'
    OFF = 'O'


# https://sumo.dlr.de/docs/TraCI/Vehicle_Signalling.html
class SumoVehSignal(object):
    """
    SumoVehSignal contains the different sumo vehicle signals.
    """
    BLINKER_RIGHT = 1 << 0
    BLINKER_LEFT = 1 << 1
    BLINKER_EMERGENCY = 1 << 2
    BRAKELIGHT = 1 << 3
    FRONTLIGHT = 1 << 4
    FOGLIGHT = 1 << 5
    HIGHBEAM = 1 << 6
    BACKDRIVE = 1 << 7
    WIPER = 1 << 8
    DOOR_OPEN_LEFT = 1 << 9
    DOOR_OPEN_RIGHT = 1 << 10
    EMERGENCY_BLUE = 1 << 11
    EMERGENCY_RED = 1 << 12
    EMERGENCY_YELLOW = 1 << 13


# https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html#abstract_vehicle_class
class SumoActorClass(enum.Enum):
    """
    SumoActorClass enumerates the different sumo actor classes.
    """
    IGNORING = "ignoring"
    PRIVATE = "private"
    EMERGENCY = "emergency"
    AUTHORITY = "authority"
    ARMY = "army"
    VIP = "vip"
    PEDESTRIAN = "pedestrian"
    PASSENGER = "passenger"
    HOV = "hov"
    TAXI = "taxi"
    BUS = "bus"
    COACH = "coach"
    DELIVERY = "delivery"
    TRUCK = "truck"
    TRAILER = "trailer"
    MOTORCYCLE = "motorcycle"
    MOPED = "moped"
    BICYCLE = "bicycle"
    EVEHICLE = "evehicle"
    TRAM = "tram"
    RAIL_URBAN = "rail_urban"
    RAIL = "rail"
    RAIL_ELECTRIC = "rail_electric"
    RAIL_FAST = "rail_fast"
    SHIP = "ship"
    CUSTOM1 = "custom1"
    CUSTOM2 = "custom2"


SumoActor = collections.namedtuple('SumoActor', 'type_id vclass transform signals extent color')

npc = collections.namedtuple('npc',['no', 'veh_type','speed','acceleration','edge','lane_length',
                                     'laneindex' ,'heading','x','y','lane_heading','lateral_pos',
                                     'FV','LFV','LFFV','LRV','LRRV','RFV','RFFV','RRV','RRRV'])
def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def calculate_lane_y(id):
    if id:
        lane=traci.vehicle.getLaneID(id) #str
        edge_no=traci.vehicle.getRoadID(id) #str
        lane_index=traci.vehicle.getLaneIndex(id) #int
        lateral_pos=traci.vehicle.getLateralLanePosition(id) #float
        i=0
        lane_y=0
        while i<lane_index:
          lane_adj=  f"{edge_no}_{i}"
          lane_y+=traci.lane.getWidth(lane_adj)
          i+=1
        lane_y+=(traci.lane.getWidth(lane)/2+lateral_pos)
        return lane_y
    else:
        return -1

def calculate_neighbour_y(id,ego_position,lane_y,lane_heading):
    if id:
        neighbour_position=traci.vehicle.getPosition(id)
        delta_x=neighbour_position[0]-ego_position[0]
        delta_y=neighbour_position[1]-ego_position[1]
        delta_l=-delta_x*math.cos(lane_heading)+delta_y*math.sin(lane_heading)
        return lane_y+delta_l
    else:
        return -1


def sumo_npc(id):
    veh_type=traci.vehicle.getTypeID(id) ##str
    lane_x=traci.vehicle.getLanePosition(id) ##float
    speed=traci.vehicle.getSpeed(id) #float
    acceleration=traci.vehicle.getAcceleration(id) #float
    lane=traci.vehicle.getLaneID(id) #str
    edge_no=traci.vehicle.getRoadID(id) #str
    edge_lanenumber=traci.edge.getLaneNumber(edge_no)
    lane_length=traci.lane.getLength(lane)##double
    lane_index=traci.vehicle.getLaneIndex(id) #int
    heading=traci.vehicle.getAngle(id)/180*math.pi #float
    lateral_pos=traci.vehicle.getLateralLanePosition(id) #float
    lane_y=calculate_lane_y(id) ##float
    coordinate=traci.vehicle.getPosition(id)
    roadpoints=traci.lane.getShape(lane)
    # 计算所有点到当前坐标的距离，并存储为 (距离, 点位) 的元组
    distances = [(calculate_distance(coordinate, roadpoints[i]), i, roadpoints[i]) for i in range(len(roadpoints))]
    # 按距离升序排序
    distances.sort()

    # 获取距离最近的两个点
    closest_two_points = distances[:2]
    closest_two_points = sorted(closest_two_points, key=lambda x: x[1])
    # print(closest_two_points)
    delta_y=closest_two_points[1][2][1]-closest_two_points[0][2][1]
    delta_x=closest_two_points[1][2][0]-closest_two_points[0][2][0]
    lane_heading = math.atan2(delta_y, delta_x) * 180 / math.pi
    if delta_x>0:
        lane_heading=90-lane_heading
    elif delta_x<0 and delta_y<0:
        lane_heading=90-lane_heading
    elif delta_x<0 and delta_y>=0:
        lane_heading=450-lane_heading

    lane_heading=lane_heading/180*math.pi
    if lane_heading>math.pi:
        lane_heading=lane_heading-math.pi
    '''跨edge情况下是否存在问题'''
    FVid = traci.vehicle.getLeader(id)[0] if traci.vehicle.getLeader(id) else False
    FVgap= traci.vehicle.getLeader(id)[1]+ traci.vehicle.getMinGap(id)+traci.vehicle.getLength(FVid) if FVid else 10000
    FVspeed= traci.vehicle.getSpeed(FVid) if FVid else -1
    FVy=calculate_neighbour_y(FVid,coordinate,lane_y,lane_heading)
    FVacc= traci.vehicle.getAcceleration(FVid) if FVid else False
    FV=[FVid,FVgap,FVspeed,FVy,FVacc]
    
    LFVid = traci.vehicle.getNeighbors(id,0b00000010)[0][0] if traci.vehicle.getNeighbors(id,0b00000010) else False
    LFVgap = traci.vehicle.getNeighbors(id,0b00000010)[0][1]+traci.vehicle.getMinGap(id) +traci.vehicle.getLength(LFVid) if LFVid else 10000
    LFVspeed = traci.vehicle.getSpeed(LFVid) if LFVid else -1
    LFVy=calculate_neighbour_y(LFVid,coordinate,lane_y,lane_heading)
    LFVacc= traci.vehicle.getAcceleration(LFVid) if LFVid else False
    LFV=[LFVid,LFVgap,LFVspeed,LFVy,LFVacc]
    
    LRVid = traci.vehicle.getNeighbors(id,0b00000000)[0][0] if traci.vehicle.getNeighbors(id,0b00000000) else False
    LRVgap = traci.vehicle.getNeighbors(id,0b00000000)[0][1]+traci.vehicle.getMinGap(LRVid)+traci.vehicle.getLength(id) if LRVid else -10000
    LRVspeed = traci.vehicle.getSpeed(LRVid) if LRVid else -1
    LRVy=calculate_neighbour_y(LRVid,coordinate,lane_y,lane_heading)
    LRVacc= traci.vehicle.getAcceleration(LRVid) if LRVid else False
    LRV=[LRVid,LRVgap,LRVspeed,LRVy,LRVacc]
    
    RFVid = traci.vehicle.getNeighbors(id,0b00000011)[0][0] if traci.vehicle.getNeighbors(id,0b00000011) else False
    RFVgap = traci.vehicle.getNeighbors(id,0b00000011)[0][1]+traci.vehicle.getMinGap(id)+traci.vehicle.getLength(RFVid)  if RFVid else 10000
    RFVspeed = traci.vehicle.getSpeed(RFVid) if RFVid else -1
    RFVy=calculate_neighbour_y(RFVid,coordinate,lane_y,lane_heading)
    RFVacc= traci.vehicle.getAcceleration(RFVid) if RFVid else False
    RFV=[RFVid,RFVgap,RFVspeed,RFVy,RFVacc]
    
    RRVid = traci.vehicle.getNeighbors(id,0b00000001)[0][0] if traci.vehicle.getNeighbors(id,0b00000001) else False
    RRVgap = traci.vehicle.getNeighbors(id,0b00000001)[0][1]+traci.vehicle.getMinGap(RRVid)+traci.vehicle.getLength(id) if RRVid else -10000
    RRVspeed = traci.vehicle.getSpeed(RRVid) if RRVid else -1
    RRVy=calculate_neighbour_y(RRVid,coordinate,lane_y,lane_heading)
    RRVacc= traci.vehicle.getAcceleration(RRVid) if RRVid else False
    RRV=[RRVid,RRVgap,RRVspeed,RRVy,RRVacc]
    
    LFFVid = traci.vehicle.getLeader(LFVid)[0] if LFVid and traci.vehicle.getLeader(LFVid) else False
    LFFVgap= traci.vehicle.getLeader(LFVid)[1]+ traci.vehicle.getMinGap(LFVid) + LFVgap+traci.vehicle.getLength(LFFVid) if LFFVid else 10000
    LFFVspeed= traci.vehicle.getSpeed(LFFVid) if LFFVid else -1
    LFFVy=calculate_neighbour_y(LFFVid,coordinate,lane_y,lane_heading)
    LFFVacc= traci.vehicle.getAcceleration(LFFVid) if LFFVid else False
    LFFV=[LFFVid,LFFVgap,LFFVspeed,LFFVy,LFFVacc]
    
    LRRVid = traci.vehicle.getFollower(LRVid)[0] if LRVid and traci.vehicle.getFollower(LRVid) else False
    LRRVgap= traci.vehicle.getFollower(LRVid)[1]+ traci.vehicle.getMinGap(LRRVid)+LRVgap+ traci.vehicle.getLength(LRVid)  if LRRVid else -10000
    LRRVspeed= traci.vehicle.getSpeed(LRRVid) if LRRVid else -1
    LRRVy=calculate_neighbour_y(LRRVid,coordinate,lane_y,lane_heading)
    LRRVacc= traci.vehicle.getAcceleration(LRRVid) if LRRVid else False
    LRRV=[LRRVid,LRRVgap,LRRVspeed,LRRVy,LRRVacc]
        
    RFFVid = traci.vehicle.getLeader(RFVid)[0] if RFVid and traci.vehicle.getLeader(RFVid) else False
    RFFVgap= traci.vehicle.getLeader(RFVid)[1]+ traci.vehicle.getMinGap(RFVid) + RFVgap+traci.vehicle.getLength(RFFVid) if RFFVid else 10000
    RFFVspeed= traci.vehicle.getSpeed(RFFVid) if RFFVid else -1
    RFFVy=calculate_neighbour_y(RFFVid,coordinate,lane_y,lane_heading)
    RFFVacc= traci.vehicle.getAcceleration(RFFVid) if RFFVid else False
    RFFV=[RFFVid,RFFVgap,RFFVspeed,RFFVy,RFFVacc]
    
    RRRVid = traci.vehicle.getFollower(RRVid)[0] if RRVid and traci.vehicle.getFollower(RRVid) else False
    RRRVgap= traci.vehicle.getFollower(RRVid)[1]+ traci.vehicle.getMinGap(RRRVid)+RRVgap+ traci.vehicle.getLength(RRVid) if RRRVid else -10000
    RRRVspeed= traci.vehicle.getSpeed(RRRVid) if RRRVid else -1
    RRRVy=calculate_neighbour_y(RRRVid,coordinate,lane_y,lane_heading)
    RRRVacc= traci.vehicle.getAcceleration(RRRVid) if RRRVid else False
    RRRV=[RRRVid,RRRVgap,RRRVspeed,RRRVy,RRRVacc]

    ego_npc=npc(no=id, veh_type=veh_type,speed=speed,acceleration=acceleration,edge=edge_no,lane_length=lane_length,
                laneindex=lane_index ,heading=heading, x=lane_x,y=lane_y,lane_heading=lane_heading,lateral_pos=lateral_pos
                ,FV=FV,LFV=LFV,LFFV=LFFV,LRV=LRV,LRRV=LRRV,RFV=RFV,RFFV=RFFV,RRV=RRV,RRRV=RRRV)
    
    # print("edge:",edge_no,"edge_lanenumber:",edge_lanenumber,'lane:',lane,"laneindex:",lane_index,
    #       "x:",lane_x,"y",lane_y,'lateral_pos',lateral_pos,'speed',speed,
    #       'FV',FV,'LFV',LFV,'LRV',LRV)
    
    return ego_npc

# ==================================================================================================
# -- sumo traffic lights ---------------------------------------------------------------------------
# ==================================================================================================


class SumoTLLogic(object):
    """
    SumoTLLogic holds the data relative to a traffic light in sumo.
    """
    def __init__(self, tlid, states, parameters):
        self.tlid = tlid
        self.states = states

        self._landmark2link = {}
        self._link2landmark = {}
        for link_index, landmark_id in parameters.items():
            # Link index information is added in the parameter as 'linkSignalID:x'
            link_index = int(link_index.split(':')[1])

            if landmark_id not in self._landmark2link:
                self._landmark2link[landmark_id] = []
            self._landmark2link[landmark_id].append((tlid, link_index))
            self._link2landmark[(tlid, link_index)] = landmark_id

    def get_number_signals(self):
        """
        Returns number of internal signals of the traffic light.
        """
        if len(self.states) > 0:
            return len(self.states[0])
        return 0

    def get_all_signals(self):
        """
        Returns all the signals of the traffic light.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return [(self.tlid, i) for i in range(self.get_number_signals())]

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with this traffic light.
        """
        return self._landmark2link.keys()

    def get_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        return self._landmark2link.get(landmark_id, [])


class SumoTLManager(object):
    """
    SumoTLManager is responsible for the management of the sumo traffic lights (i.e., keeps control
    of the current program, phase, ...)
    """
    def __init__(self):
        self._tls = {}  # {tlid: {program_id: SumoTLLogic}
        self._current_program = {}  # {tlid: program_id}
        self._current_phase = {}  # {tlid: index_phase}

        for tlid in traci.trafficlight.getIDList():
            self.subscribe(tlid)

            self._tls[tlid] = {}
            for tllogic in traci.trafficlight.getAllProgramLogics(tlid):
                states = [phase.state for phase in tllogic.getPhases()]
                parameters = tllogic.getParameters()
                tl = SumoTLLogic(tlid, states, parameters)
                self._tls[tlid][tllogic.programID] = tl

            # Get current status of the traffic lights.
            self._current_program[tlid] = traci.trafficlight.getProgram(tlid)
            self._current_phase[tlid] = traci.trafficlight.getPhase(tlid)

        self._off = False

    @staticmethod
    def subscribe(tlid):
        """
        Subscribe the given traffic ligth to the following variables:

            * Current program.
            * Current phase.
        """
        traci.trafficlight.subscribe(tlid, [
            traci.constants.TL_CURRENT_PROGRAM,
            traci.constants.TL_CURRENT_PHASE,
        ])

    @staticmethod
    def unsubscribe(tlid):
        """
        Unsubscribe the given traffic ligth from receiving updated information each step.
        """
        traci.trafficlight.unsubscribe(tlid)

    def get_all_signals(self):
        """
        Returns all the traffic light signals.
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_all_signals())
        return signals

    def get_all_landmarks(self):
        """
        Returns all the landmarks associated with a traffic light in the simulation.
        """
        landmarks = set()
        for tlid, program_id in self._current_program.items():
            landmarks.update(self._tls[tlid][program_id].get_all_landmarks())
        return landmarks

    def get_all_associated_signals(self, landmark_id):
        """
        Returns all the signals associated with the given landmark.
            :returns list: [(tlid, link_index), (tlid, link_index), ...]
        """
        signals = set()
        for tlid, program_id in self._current_program.items():
            signals.update(self._tls[tlid][program_id].get_associated_signals(landmark_id))
        return signals

    def get_state(self, landmark_id):
        """
        Returns the traffic light state of the signals associated with the given landmark.
        """
        states = set()
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            current_program = self._current_program[tlid]
            current_phase = self._current_phase[tlid]

            tl = self._tls[tlid][current_program]
            states.update(tl.states[current_phase][link_index])

        if len(states) == 1:
            return states.pop()
        elif len(states) > 1:
            logging.warning('Landmark %s is associated with signals with different states',
                            landmark_id)
            return SumoSignalState.RED
        else:
            return None

    def set_state(self, landmark_id, state):
        """
        Updates the state of all the signals associated with the given landmark.
        """
        for tlid, link_index in self.get_all_associated_signals(landmark_id):
            traci.trafficlight.setLinkState(tlid, link_index, state)
        return True

    def switch_off(self):
        """
        Switch off all traffic lights.
        """
        for tlid, link_index in self.get_all_signals():
            traci.trafficlight.setLinkState(tlid, link_index, SumoSignalState.OFF)
        self._off = True

    def tick(self):
        """
        Tick to traffic light manager
        """
        if self._off is False:
            for tl_id in traci.trafficlight.getIDList():
                results = traci.trafficlight.getSubscriptionResults(tl_id)
                current_program = results[traci.constants.TL_CURRENT_PROGRAM]
                current_phase = results[traci.constants.TL_CURRENT_PHASE]

                if current_program != 'online':
                    self._current_program[tl_id] = current_program
                    self._current_phase[tl_id] = current_phase


# ==================================================================================================
# -- sumo simulation -------------------------------------------------------------------------------
# ==================================================================================================

def _get_sumo_net(cfg_file):
    """
    Returns sumo net.

    This method reads the sumo configuration file and retrieve the sumo net filename to create the
    net.
    """
    cfg_file = os.path.join(os.getcwd(), cfg_file)

    tree = ET.parse(cfg_file)
    tag = tree.find('//net-file')
    if tag is None:
        return None

    net_file = os.path.join(os.path.dirname(cfg_file), tag.get('value'))
    logging.debug('Reading net file: %s', net_file)

    sumo_net = sumolib.net.readNet(net_file)
    return sumo_net


class SumoSimulation(object):
    """
    SumoSimulation is responsible for the management of the sumo simulation.
    """
    def __init__(self, agent,cfg_file, step_length, seed,host=None, port=None, sumo_gui=False, client_order=1):
        if sumo_gui is True:
            sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            sumo_binary = sumolib.checkBinary('sumo')

        if host is None or port is None:
            logging.info('Starting new sumo server...')
            if sumo_gui is True:
                logging.info('Remember to press the play button to start the simulation')

            traci.start([sumo_binary,
                '--seed', str(seed),         
                '--configuration-file', cfg_file,
                '--step-length', str(step_length),
                '--lateral-resolution', '0.25',
                '--collision.check-junctions'
            ])

        else:
            logging.info('Connection to sumo server. Host: %s Port: %s', host, port)
            traci.init(host=host, port=port)

        traci.setOrder(client_order)   
        # Retrieving net from configuration file.
        self.net = _get_sumo_net(cfg_file)
        # To keep track of the vehicle classes for which a route has been generated in sumo.
        self._routes = set()

        # Variable to asign an id to new added actors.
        self._sequential_id = 0

        # Structures to keep track of the spawned and destroyed vehicles at each time step.
        self.spawned_actors = set()
        self.destroyed_actors = set()

        # Traffic light manager.
        self.traffic_light_manager = SumoTLManager()

        self.timestep=step_length
        self.compete_id='0'
        self.compete_vehicle_flag=False
        self.ego_id = '10000'
        self.egovehicle_flag= False
        self.is_vdm=True
        self.agent=agent
        self.generate_time=random.randint(6,12)/10 #加塞场景对手车生成时刻

    @property
    def traffic_light_ids(self):
        return self.traffic_light_manager.get_all_landmarks()

    @staticmethod
    def subscribe(actor_id):
        """
        Subscribe the given actor to the following variables:

            * Type.
            * Vehicle class.
            * Color.
            * Length, Width, Height.
            * Position3D (i.e., x, y, z).
            * Angle, Slope.
            * Speed.
            * Lateral speed.
            * Signals.
        """
        traci.vehicle.subscribe(actor_id, [
            traci.constants.VAR_TYPE, traci.constants.VAR_VEHICLECLASS, traci.constants.VAR_COLOR,
            traci.constants.VAR_LENGTH, traci.constants.VAR_WIDTH, traci.constants.VAR_HEIGHT,
            traci.constants.VAR_POSITION3D, traci.constants.VAR_ANGLE, traci.constants.VAR_SLOPE,
            traci.constants.VAR_SPEED, traci.constants.VAR_SPEED_LAT, traci.constants.VAR_SIGNALS
        ])

    @staticmethod
    def unsubscribe(actor_id):
        """
        Unsubscribe the given actor from receiving updated information each step.
        """
        traci.vehicle.unsubscribe(actor_id)

    def get_net_offset(self):
        """
        Accessor for sumo net offset.
        """
        if self.net is None:
            return (0, 0)
        return self.net.getLocationOffset()

    @staticmethod
    def get_actor(actor_id):
        """
        Accessor for sumo actor.
        """
        results = traci.vehicle.getSubscriptionResults(actor_id)

        type_id = results[traci.constants.VAR_TYPE]
        vclass = SumoActorClass(results[traci.constants.VAR_VEHICLECLASS])
        color = results[traci.constants.VAR_COLOR]

        length = results[traci.constants.VAR_LENGTH]
        width = results[traci.constants.VAR_WIDTH]
        height = results[traci.constants.VAR_HEIGHT]

        location = list(results[traci.constants.VAR_POSITION3D])
        rotation = [results[traci.constants.VAR_SLOPE], results[traci.constants.VAR_ANGLE], 0.0]
        transform = carla.Transform(carla.Location(location[0], location[1], location[2]),
                                    carla.Rotation(rotation[0], rotation[1], rotation[2]))

        signals = results[traci.constants.VAR_SIGNALS]
        extent = carla.Vector3D(length / 2.0, width / 2.0, height / 2.0)

        # if actor_id=='10000':
        #     print(location)
        #     print(rotation)
        return SumoActor(type_id, vclass, transform, signals, extent, color)

    def spawn_actor(self, type_id, color=None):
        """
        Spawns a new actor.

            :param type_id: vtype to be spawned.
            :param color: color attribute for this specific actor.
            :return: actor id if the actor is successfully spawned. Otherwise, INVALID_ACTOR_ID.
        """
        actor_id = 'carla' + str(self._sequential_id)
        try:
            vclass = traci.vehicletype.getVehicleClass(type_id)
            if vclass not in self._routes:
                logging.debug('Creating route for %s vehicle class', vclass)
                allowed_edges = [e for e in self.net.getEdges() if e.allows(vclass)]
                if allowed_edges:
                    traci.route.add("carla_route_{}".format(vclass), [allowed_edges[0].getID()])
                    self._routes.add(vclass)
                else:
                    logging.error(
                        'Could not found a route for %s. No vehicle will be spawned in sumo',
                        type_id)
                    return INVALID_ACTOR_ID

            traci.vehicle.add(actor_id, 'carla_route_{}'.format(vclass), typeID=type_id)
        except traci.exceptions.TraCIException as error:
            logging.error('Spawn sumo actor failed: %s', error)
            return INVALID_ACTOR_ID

        if color is not None:
            color = color.split(',')
            traci.vehicle.setColor(actor_id, color)

        self._sequential_id += 1

        return actor_id

    @staticmethod
    def destroy_actor(actor_id):
        """
        Destroys the given actor.
        """
        traci.vehicle.remove(actor_id)

    def get_traffic_light_state(self, landmark_id):
        """
        Accessor for traffic light state.

        If the traffic ligth does not exist, returns None.
        """
        return self.traffic_light_manager.get_state(landmark_id)

    def switch_off_traffic_lights(self):
        """
        Switch off all traffic lights.
        """
        self.traffic_light_manager.switch_off()

    def synchronize_vehicle(self, vehicle_id, transform, signals=None):
        """
        Updates vehicle state.

            :param vehicle_id: id of the actor to be updated.
            :param transform: new vehicle transform (i.e., position and rotation).
            :param signals: new vehicle signals.
            :return: True if successfully updated. Otherwise, False.
        """
        loc_x, loc_y = transform.location.x, transform.location.y
        yaw = transform.rotation.yaw

        traci.vehicle.moveToXY(vehicle_id, "", 0, loc_x, loc_y, angle=yaw, keepRoute=2)
        if signals is not None:
            traci.vehicle.setSignals(vehicle_id, signals)
        return True

    def synchronize_traffic_light(self, landmark_id, state):
        """
        Updates traffic light state.

            :param tl_id: id of the traffic light to be updated (logic id, link index).
            :param state: new traffic light state.
            :return: True if successfully updated. Otherwise, False.
        """
        self.traffic_light_manager.set_state(landmark_id, state)

    def safety_checker(self):
        # print('safety checker')
        ACC_prmts= [17,1.0,4,2,3 ] 
        ss = 0
        spd_temp = 0
        zs = 0
        a_free = 0
        desiredSpeed = 30/3.6 ##30,60,90
        Speed=self.competevehcile.speed
        LeadingVeh_Dist=self.competevehcile.FVdis-self.competevehcile.mingap-self.competevehcile.length
        LeadingVeh_Spd = self.competevehcile.FVspeed

        if (desiredSpeed < LeadingVeh_Spd):
            spd_temp = desiredSpeed
        else:
            spd_temp = LeadingVeh_Spd
        ss = ACC_prmts[2] + max(0, (Speed * ACC_prmts[1] + (Speed * (Speed - spd_temp)) / (2 * math.sqrt(ACC_prmts[3] * ACC_prmts[4]))))
        #异常数值处理
        if (LeadingVeh_Dist <= 0):                
            LeadingVeh_Dist = 0.1
        zs = ss / LeadingVeh_Dist
        ##异常数值处理
        if(zs<0):                
            zs =0                
        if (Speed <= desiredSpeed):
            a_free = ACC_prmts[3] * (1 - pow(Speed / desiredSpeed, 4))
            if (zs >= 1):
                desiredacc=ACC_prmts[3] * (1 - zs**2)
            else:                    
                if (a_free ==0):                        
                    desiredacc=0                        
                else:                        
                    desiredacc=a_free * (1 - pow(zs, 2 * ACC_prmts[3] / a_free))                                   
        else:
            a_free = -ACC_prmts[4] * (1 - pow(desiredSpeed / Speed, ACC_prmts[3] * 4 / ACC_prmts[4]))
            if (zs >= 1):
                desiredacc=a_free + ACC_prmts[3] * (1 - pow(zs, 2))
            else:
                desiredacc=a_free
        if (desiredacc > 2):                
            desiredacc=2
        
        if (desiredacc< -6): 
            desiredacc=-6   
        return desiredacc
    
    def get_vehicle_info(self):
        id_list = traci.vehicle.getIDList()
        if self.ego_id in id_list:
            ego_vehicle_info=sumo_npc(self.ego_id)
        else:
            ego_vehicle_info=False
        return ego_vehicle_info
        
    def tick(self,control,compete_target):
        """
        Tick to sumo simulation.
        """
        '''ego vehicle'''
        time=traci.simulation.getTime()
        if time == 30:
            ##生成车辆
            traci.vehicle.add(vehID=self.ego_id,routeID="r2",typeID="vehicle.tesla.model3",depart=time,departLane="2",departSpeed="desired")
            traci.vehicle.highlight(self.ego_id) #将该辆车在路网中标记出来
            # traci.gui.trackVehicle("View #0", self.ego_id) #将仿真视角跟随标记的车辆，"View #0"为SUMO默认的视角名称
            traci.vehicle.setLaneChangeMode(self.ego_id,0)
            traci.vehicle.setSpeedMode(self.ego_id,0)
        

        '''被加塞车辆'''
        if time==30-self.generate_time:
            traci.vehicle.add(vehID='10001',routeID="r2",typeID="vehicle.tesla.model3",depart=time,departLane="3",departSpeed="desired")
        
        if time==30 + self.generate_time:
            traci.vehicle.changeLane('10001',2,4*self.generate_time)
            
        
        id_list = traci.vehicle.getIDList()

        if control["acc"]:
            coordinate=traci.vehicle.getPosition(self.ego_id)
            traci.vehicle.setAcceleration(self.ego_id,2*control['acc'],self.timestep)
            # print(traci.vehicle.getSpeed(self.ego_id),traci.vehicle.getAcceleration(self.ego_id))
            move_x=coordinate[0]+control["delta_x"]
            move_y=coordinate[1]+control["delta_y"]
            yaw=traci.vehicle.getAngle(self.ego_id)+control["delta_fi"]
            # print(traci.vehicle.getAngle(self.ego_id),control["delta_fi"])
            traci.vehicle.moveToXY(self.ego_id, "", 0, move_x, move_y, angle=yaw, keepRoute=2)
        
        """
        select the vehicle //generate the vehicle
        get vehcile state
        PPO model import
        model step
        remove vdm
        """
        '''指定对手车调用rl模型'''
        if compete_target!=False:
            compete_id=compete_target[0]
            if compete_id!=self.compete_id:
                ##新的对手车
                if self.compete_vehicle_flag==False:
                    self.compete_vehicle_flag==True
                elif self.compete_vehicle_flag and self.is_vdm:
                    traci.vehicle.setColor(self.compete_id,(255,0, 0,255))
                    self.competevehcile.vdm._Z16toyota_terminatev()
                    libc = ctypes.CDLL(None)  # 加载C标准库
                    libc.dlclose(self.competevehcile.vdm._handle)  # 关闭动态库
                self.compete_id=compete_id
                self.competevehcile = Vehicle(self.compete_id,self.is_vdm)
                # traci.vehicle.setLaneChangeMode(self.compete_id,0)
                traci.vehicle.setSpeedMode(self.compete_id,0)
                traci.vehicle.setColor(self.compete_id,(0, 255, 0, 255))
            else:
                self.competevehcile.update()
                
            state=[self.competevehcile.speed,self.competevehcile.acceleration,
                    self.competevehcile.FVdis,self.competevehcile.FVspeed-self.competevehcile.speed,
                    self.competevehcile.LFVdis,self.competevehcile.LFVspeed-self.competevehcile.speed,
                    self.competevehcile.RFVdis,self.competevehcile.RFVspeed-self.competevehcile.speed]
                
            action = self.agent.evaluate(state)
            desiredacc=action[0]*4-2 #-6~2
            if self.is_vdm:
                ##safety checker
                if self.competevehcile.speed>0 and (self.competevehcile.FVdis-self.competevehcile.mingap-self.competevehcile.length)/self.competevehcile.speed<0.5:
                    desiredacc=self.safety_checker()    
                desiredacc_vdm=self.competevehcile.vdm._Z11toyota_stepdd(desiredacc,self.competevehcile.initial_speed)
                desired_speed=self.competevehcile.speed+desiredacc_vdm*self.timestep
                # print(round(desiredacc,2),'  ',round(desiredacc_vdm,2))
                traci.vehicle.setAcceleration(self.compete_id,2*desiredacc_vdm,self.timestep)
                traci.vehicle.setSpeed(self.compete_id,desired_speed)
            else:
                if self.competevehcile.speed>0 and (self.competevehcile.FVdis-self.competevehcile.mingap-self.competevehcile.length)/self.competevehcile.speed<0.5:
                    desiredacc=self.safety_checker()   
                desired_speed=self.competevehcile.speed+desiredacc*self.timestep
                traci.vehicle.setAcceleration(self.compete_id,2*desiredacc,self.timestep)
                traci.vehicle.setSpeed(self.compete_id,desired_speed)
              
        traci.simulationStep()
        self.traffic_light_manager.tick()

        # Update data structures for the current frame.
        self.spawned_actors = set(traci.simulation.getDepartedIDList())
        self.destroyed_actors = set(traci.simulation.getArrivedIDList())
        

    @staticmethod
    def close():
        """
        Closes traci client.
        """
        traci.close()
