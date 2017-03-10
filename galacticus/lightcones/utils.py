#! /usr/bin/env python

import numpy as np
from ..constants import Pi


def getRaDec(lightconePositionX,lightconePositionY,lightconePositionZ,degrees=True):    
    R = np.sqrt(lightconePositionX**2+lightconePositionY**2+lightconePositionZ**2)
    rightAscension = np.arctan2(lightconePositionY,lightconePositionX)
    declination = np.arcsin(lightconePositionZ/R)
    if degrees:
        rightAscension *= (180.0/Pi)
        declination *= (180.0/Pi)
    return rightAscension,declination
