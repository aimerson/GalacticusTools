#! /usr/bin/env python

import sys
import numpy as np


class KitzbichlerWhite2007(object):
    
    def __init__(self,boxLength,boxUnits="Mpc/h"):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.boxLength = boxLength
        self.boxUnits = boxUnits
        self.m = None
        self.n = None
        return

    def computeIntegers(self,fieldSize,verbose=True):
        # Compute the values of m and n. Here we assume n=m+1 (to keep
        # n and m as small as possible. According to Kitzbichler &
        # White (2007; MNRAS; 376; 2) this will give a field of
        # dimensions 1/m^2/n by 1/m/n^2 radians. Given the inverse
        # anglular size of the survey "invAngle" we can solve the
        # resulting cubic equation for m. Solution taken from Andrew
        # Benson's script Millennium_Lightcone_Grab.pl.
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        field = np.radians(fieldSize)
        invAngle = 1.0/field
        m = (8.0+108.0*invAngle+12.0*np.sqrt(12.0*invAngle+81.0*invAngle**2))**(1.0/3.0)/6.0 + \
            2.0/3.0*(8.0+108.0*invAngle+12.0*np.sqrt(12.0*invAngle+81.0*invAngle**2))**(-1.0/3.0)-2.0/3.0
        m = int(m)
        if m < 1 :
            raise ValueError(funcname+"(): Lightcone angle too large!")
        else:
            self.m = int(m)
            self.n = self.m + 1            
        if verbose:
            print(funcname+"(): Integer divisors (m,n) = ("+str(self.m)+", "+str(self.n)+")")
            self.computeAngles(fieldSize,verbose=verbose)
        return

    def computeAngles(self,fieldSize,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.m is None:
            self.computeIntegers(fieldSize)
        else:
            self.n = self.m + 1
        self.angle1 = np.degrees(1.0/float(self.m)/float(self.n)/float(self.n))
        self.angle2 = np.degrees(1.0/float(self.m)/float(self.m)/float(self.n))
        if verbose:
            print(funcname+"(): Field is "+str(self.angle1)+" x "+str(self.angle2)+" degrees")
        return

    def maxDepth(self,fieldSize,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.m is None:
            self.computeIntegers(fieldSize)
        else:
            self.n = self.m + 1
        self.maxDepth = np.sqrt(float(self.n**2+self.m**2+(self.n*self.m)**2))*self.boxLength
        if verbose:
            print(funcname+"(): Maximum depth before repeats = "+str(self.maxDepth)+" "+self.boxUnits)
        return



