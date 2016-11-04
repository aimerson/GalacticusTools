#! /usr/bin/env python

import numpy as np

def getDataType(arr):
    dtype = str(arr.dtype)
    dtype = dtype.replace("float32","float64")
    return dtype

