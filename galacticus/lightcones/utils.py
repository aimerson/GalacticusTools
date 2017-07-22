#! /usr/bin/env python

import numpy as np
from ..constants import Pi
from ..cosmology import Cosmology


def getRaDec(lightconePositionX,lightconePositionY,lightconePositionZ,degrees=True):    
    R = np.sqrt(lightconePositionX**2+lightconePositionY**2+lightconePositionZ**2)
    rightAscension = np.arctan2(lightconePositionY,lightconePositionX)
    declination = np.arcsin(lightconePositionZ/R)
    if degrees:
        rightAscension *= (180.0/Pi)
        declination *= (180.0/Pi)
    return rightAscension,declination

def getCartesianXYZ(lightconeRA,lightconeDec,lightconeRedshift,cosmologyObj,degrees=True):
    distance = cosmologyObj.comoving_distance(lightconeRedshift)
    if degrees:
        lightconeRA *= (Pi/180.0)
        lightconeDec *= (Pi/180.0)
    X = distance*np.cos(lightconeDec)*np.cos(lightconeRA)
    Y = distance*np.cos(lightconeDec)*np.sin(lightconeRA)
    Z = distance*np.sin(lightconeDec)
    return X,Y,Z
