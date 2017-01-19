#! /usr/bin/env python

import numpy as np


class TemporaryClass(object):
    
    def __init__(self,array):
        self.array = np.copy(array)
        return

    def updateArray(self,newArray,updateFunction):
        self.array = updateFunction(self.array,np.copy(newArray))
        return
        
        
