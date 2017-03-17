#! /usr/bin/env python

import numpy as np


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


class TemporaryClass(object):
    
    def __init__(self,array):
        self.array = np.copy(array)
        return

    def updateArray(self,newArray,updateFunction):
        self.array = updateFunction(self.array,np.copy(newArray))
        return
        
        
