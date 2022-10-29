import numpy as np
from os import path
import math

import torch
import random
import math
import torch.nn as nn
from RBC_env import RBC
simulator = RBC()
simulator.reset(ctr=0.1, const=1.0)
simulator.solve()
