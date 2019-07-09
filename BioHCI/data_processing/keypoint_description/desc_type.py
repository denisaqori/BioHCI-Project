"""
Created: 5/21/19
© Denisa Qori McDonald 2019 All Rights Reserved
"""
from enum import Enum, unique


@unique
class DescType(Enum):
    RawData = 0
    JUSD = 1
    MSBSD = 2
