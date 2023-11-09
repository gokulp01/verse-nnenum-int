from enum import Enum, auto
import copy
import random
import numpy as np
from typing import List
import logging

# Configure logging at the start of your script.
# Set up logging to file
logging.basicConfig(filename='myapp.log', level=logging.INFO)

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



    def __init__(self, x, y, theta, random_attr, mode, velo):
        pass

def decisionLogic(ego: State, others: List[State]):
    next = copy.deepcopy(ego)

    if ego.mode == CraftMode.Coc:

        if ego.x > 2000:
            next.mode = CraftMode.Strong_right




    return next


