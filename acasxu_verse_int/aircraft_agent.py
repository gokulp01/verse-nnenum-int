# aircraft agent.
from typing import Tuple, List

import numpy as np
from scipy.integrate import ode
import torch

# from tutorial_utils import drone_params
from verse import BaseAgent
from verse import LaneMap
from verse.map.lane_map_3d import LaneMap_3d
from verse.analysis.utils import wrap_to_pi
from verse.analysis.analysis_tree import TraceType



from functools import lru_cache
import time
import math
import argparse

from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
from verse.parser.parser import ControllerIR

import onnxruntime as ort
from numba import njit



class AircraftAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None,
        velo=None,
    ):
        super().__init__(
            id, code, file_name, initial_state=initial_state, initial_mode=initial_mode
        )
        self.velo = velo
        # self.decision_logic = ControllerIR.empty()



    def action_handler(self, mode: str, state, lane_map: LaneMap) -> int:
        x1, y1, theta1, _ = state
        ego_mode = mode[0]
        a1=0
        # print(ego_mode)
        # print("kkk")
        if ego_mode == "Coc":
            a1= 0
        elif ego_mode == "Weak_left":
            a1= 1
        elif ego_mode == "Weak_right":
            a1= 2
        elif ego_mode == "Strong_left":
            a1= 3
        elif ego_mode == "Strong_right":
            a1= 4
        else:
            raise ValueError(f"Invalid mode: {ego_mode}")
        
        # if int_mode == "coc":
        #     a2= 0
        # elif int_mode == "weak_left":
        #     a2= 1
        # elif int_mode == "weak_right":
        #     a2= 2
        # elif int_mode == "strong_left":
        #     a2= 3
        # elif int_mode == "strong_right":
        #     a2= 4
        # else:
        #     raise ValueError(f"Invalid mode: {int_mode}")
        return a1

    def TC_simulate(
        self, mode: str, init, time_bound, time_step, lane_map: LaneMap = None
    ) -> np.ndarray:
        time_bound = float(time_bound)
        num_points = int(np.ceil(time_bound / time_step))
        trace = np.zeros((num_points + 1, 1 + len(init)))
        trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        trace[0, 1:] = init
        vec=init
        dt_acas=1.0
        time_elapse_mats = init_time_elapse_mats(dt_acas)
        for i in range(num_points):
            # print(mode)
            cmd = self.action_handler(mode, init, lane_map)
            time_elapse_mat = time_elapse_mats[0][cmd] #get_time_elapse_mat(self.command, State.dt, intruder_cmd)
            vec = step_state(vec, self.velo, time_elapse_mat, dt_acas)
            # print(vec[1])

            init = vec.flatten()
            trace[i + 1, 0] = time_step * (i + 1)
            trace[i + 1, 1:] = init
        return trace

def step_state(state7, v, time_elapse_mat, dt):
    """perform one time step with the given commands"""
    state8_vec = state4_to_state5(state7, v)
   

    s = time_elapse_mat @ state8_vec
    # extract observation (like theta) from state
    new_time = state7[-1] + dt
    theta1 = math.atan2(s[3], s[2])
    rv = np.array([s[0], s[1], theta1, new_time])

    return rv
def state4_to_state5(state7, v):
    """compute x,y, vx, vy from state7"""

    assert len(state7) == 4

    x1 = state7[0]
    y1 = state7[1]
    vx1 = math.cos(state7[2]) * v
    vy1 = math.sin(state7[2]) * v

    return np.array([x1, y1, vx1, vy1])

'Lru Cache.'
def get_time_elapse_mat(command, dt):
    '''get the matrix exponential for the given command

    state: x, y, vx, vy
    '''

    y_list = [0.0, 1.5, -1.5, 3.0, -3.0]
    y1 = y_list[command]
    # y2 = y_list[command2]

    dtheta1 = (y1 / 180 * np.pi)
    # dtheta2 = (y2 / 180 * np.pi)
    a_mat = np.array([
        [0, 0, 1, 0], # x' = vx
        [0, 0, 0, 1], # y' = vy
        [0, 0, 0, -dtheta1], # vx' = -vy * dtheta1
        [0, 0, dtheta1, 0], # vy' = vx * dtheta1
        ], dtype=float)
    return expm(a_mat * dt)


def init_time_elapse_mats(dt):
    """get value of time_elapse_mats array"""

    rv = []
    rv.append([]) 
    for cmd in range(5):
           

        mat = get_time_elapse_mat(cmd, dt)
        rv[-1].append(mat)

    return rv




# rough
# init_vec, cmd_list, init_velo = make_random_input(interesting_seed=671, intruder_can_turn=False)
#         self.x1 = init_vec[0]
#         self.y1 = init_vec[1]
#         self.theta1 = init_vec[2]
#         self.x2 = init_vec[3]
#         self.y2 = init_vec[4]
#         self.theta2 = init_vec[5]
#         self.random_attr = init_vec[6]
#         self.ego_velo = init_velo[0]
#         self.int_velo = init_velo[1]
#         self.ego_mode = CraftMode.coc
#         self.int_mode = CraftMode.coc
