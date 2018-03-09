#! /usr/bin/env python

def getHostIndex(nodeIsIsolated):
    nodes = nodeIsIsolated[::-1]
    def getIndex(values,i):
        global previous
        if values[i] == 1:
            previous = len(values)-i-1
            return previous
        result = np.array([getIndex(nodes,i)for i in range(len(nodes))])
    result = result[::-1]
    return result






    
