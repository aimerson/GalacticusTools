#! /usr/bin/env python                                                                                                                                                                                                      

import time

def time_function(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        dt = t2 - t1
        if dt < 1.00:
            print('{0:s} took {1:0.3f} ms'.format(func.func_name,dt*1000.0))
        elif dt >= 1.00  and dt < 60.0:
            print('{0:s} took {1:0.3f} s'.format(func.func_name,dt))
        elif dt >= 60.0 and dt <= 3600.0:
            dt = dt/60.0
            print('{0:s} took {1:0.3f} min'.format(func.func_name,dt))
        else:
            dt = dt/(60.0*60.0)
            print('{0:s} took {1:0.3f} hr'.format(func.func_name,dt))
        return res
    return wrapper


class STOPWATCH:

    def __init__(self):
        classname = self.__class__.__name__
        self.t = time.time()
        print(classname+"(): Starting clock...")
        return

    def stop(self):
        classname = self.__class__.__name__
        dt = time.time() - self.t
        if dt < 1.00:
            value = 'time elapsed = {0:0.3f} ms'.format(dt*1000.0)
        elif dt >= 1.00  and dt < 60.0:
            value = 'time elapsed = {0:0.3f} s'.format(dt)
        elif dt >= 60.0 and dt <= 3600.0:
            dt = dt/60.0
            value =  'time elapsed = {0:0.3f} min'.format(dt)
        else:
            dt = dt/(60.0*60.0)
            value =  'time elapsed = {0:0.3f} hr'.format(dt)
        print(classname+"(): "+value)
        return
