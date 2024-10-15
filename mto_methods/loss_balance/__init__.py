# import torch
# import torch.nn.functional as F
# from typing import Union, List

from .famo import FAMO
from .uncertainty import Uncertainty
from .base import ScaleInvariantLinearScalarization,STL,LinearScalarization,RLW
from .dynamic_weight_average import DynamicWeightAverage