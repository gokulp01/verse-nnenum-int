# %%
import verse

# %%
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

from numba import njit

# %%

def make_random_input(seed, intruder_can_turn=True, num_inputs=100):
    """make a random input for the system"""

    np.random.seed(seed) # deterministic random numbers

    # state vector is: x, y, theta, x2, y2, theta2, time
    init_vec = np.zeros(7)
    init_vec[2] = np.pi / 2 # ownship moving up initially

    radius = 10000 + np.random.random() * 55000 # [10000, 65000]
    angle = np.random.random() * 2 * np.pi
    int_x = radius * np.cos(angle)
    int_y = radius * np.sin(angle)
    int_heading = np.random.random() * 2 * np.pi

    init_vec[3] = int_x
    init_vec[4] = int_y
    init_vec[5] = int_heading

    # intruder commands for every control period (0 to 4)
    if intruder_can_turn:
        cmd_list = []

        for _ in range(num_inputs):
            cmd_list.append(np.random.randint(5))
    else:
        cmd_list = [0] * num_inputs

    # generate random valid velocities
    #init_velo = [np.random.randint(100, 1146),
    #             np.random.randint(60, 1146)]
    init_velo = [np.random.randint(100, 1200),
                 np.random.randint(0, 1200)]

    return init_vec, cmd_list, init_velo


# %%
from dl_acas import CraftMode

init_vec, cmd_list, init_velo = make_random_input(seed=671, intruder_can_turn=False)
x1 = init_vec[0]
y1 = init_vec[1]
theta1 = init_vec[2]
x2 = init_vec[3]
y2 = init_vec[4]
theta2 = init_vec[5]
random_attr = init_vec[6]
ego_velo = init_velo[0]
int_velo = init_velo[1]
ego_mode = CraftMode.Coc
int_mode = CraftMode.Coc


# %%
decisions = np.load('commands_671.npy')
decisions = decisions.tolist()

# %%
from aircraft_agent import AircraftAgent



# %%
from verse.scenario import Scenario
from verse.scenario import ScenarioConfig

from tutorial_map import M4

scenario = Scenario()
# scenario.set_map(M4())

# %%
ac1 = AircraftAgent("aircraft1", file_name="dl_acas.py", initial_mode=ego_mode, velo=ego_velo)

ac1.set_initial(
    [[-0.5, -0.5, -0.5, 0.0], [0.5, 0.5, 0.5, 0.0]], ([CraftMode.Strong_left]),
)
ac2 = AircraftAgent("aircraft2", file_name="dl_acas.py", initial_mode=int_mode, velo=int_velo)

ac2.set_initial(
    [[9, 9, 9, 0.0], [10, 10, 10, 0.0]], ([CraftMode.Coc]),
)

scenario.add_agent(ac1)
scenario.add_agent(ac2)

# %%
from tutorial_sensor import DefaultSensor

scenario.set_sensor(DefaultSensor())

# %%
traces_simu = scenario.simulate(60, 0.2)
traces_veri = scenario.verify(60, 0.2)

# %%


# %%
from verse.plotter.plotter3D import *
import pyvista as pv
import warnings

warnings.filterwarnings("ignore")

from verse.plotter.plotter2D import *
import plotly.graph_objects as go

fig = go.Figure()
fig = reachtube_tree(traces_veri, None, fig, 1, 3)
fig.show()

pv.set_jupyter_backend(None)
fig = pv.Plotter()
fig = plot3dMap(M4(), ax=fig)
fig = plot3dReachtube(traces_veri, "aircraft1", 1, 2, 3, color="r", ax=fig)
# fig = plot3dReachtube(traces_veri, "aircraft2", 1, 2, 3, color="b", ax=fig)
fig.set_background("#e0e0e0")
fig.show()

# %%



