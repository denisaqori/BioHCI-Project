"""
Created: 7/26/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""

from enum import Enum, unique


@unique
class SequenceLength(Enum):
    Undersample = -1
    Existing = 0
    ZeroPad = 1
    ExtendEdge = 2


