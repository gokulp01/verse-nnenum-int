from enum import Enum, auto
import copy
import random
import numpy as np
from typing import List
import logging

# Configure logging at the start of your script.
# Set up logging to file
# logging.basicConfig(filename='myapp.log', level=logging.INFO)

class CraftMode(Enum):
    Coc = auto()
    Weak_left = auto()
    Weak_right = auto()
    Strong_left = auto()
    Strong_right = auto()


# class TrackMode(Enum):
#     T0 = auto()
#     T1 = auto()
#     T2 = auto()
#     T3 = auto()
#     T4 = auto()


class State:
    x: float
    y: float
    theta: float
    random_attr: float
    velo: float
    mode: CraftMode



    def __init__(self, x, y, theta, random_attr, velo, mode):
        pass

def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)
    if ego.mode == CraftMode.Coc:
        if ego.x > 0:
            next.mode = CraftMode.Weak_left
    # if cmd==0:
    #     next.mode = CraftMode.Coc
    # elif cmd==1:
    #     next.mode = CraftMode.Weak_left
    # elif cmd==2:
    #     next.mode = CraftMode.Weak_right
    # elif cmd==3:
    #     next.mode = CraftMode.Strong_left
    # elif cmd==4:
    #     next.mode = CraftMode.Strong_right




    return next


