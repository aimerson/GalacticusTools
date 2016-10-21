#! /usr/bin/env python

import sys
import numpy as np
import xml.etree.ElementTree as ET
from ..simulations import Simulation
from ..parameters import formatParametersFile
from .footprint import KitzbichlerWhite2007



class Geometry(object):
    
    def __init__(self,treeRoot):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.root = treeRoot
        return



class buildGeometry(Geometry):
    
    def __init__(self,simulationName):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Intialise simulation object
        self.simulation = Simulation(simulationName)
        # Initialise XML tree
        root = ET.Element("geometry")
        # Initalise Geometry class
        super(buildGeometry,self).__init__(root)
        # Add simulation information
        ET.SubElement(self.root,"boxLength").text = self.simulation.boxSize
        COS = ET.SubElement(self.root,"cosmology")    
        PARAM = ET.SubElement(COS,"parameter")    
        ET.SubElement(PARAM,"name").text = "omega0"
        ET.SubElement(PARAM,"value").text = str(self.simulation.omega0)
        PARAM = ET.SubElement(COS,"parameter")    
        ET.SubElement(PARAM,"name").text = "lambda0"
        ET.SubElement(PARAM,"value").text = str(self.simulation.lambda0)
        PARAM = ET.SubElement(COS,"parameter")    
        ET.SubElement(PARAM,"name").text = "H0"
        if "h" in self.simulation.boxSizeUnits:
            ET.SubElement(PARAM,"value").text = "100"
        else:
            ET.SubElement(PARAM,"value").text = str(self.simulation.h0*100.0)        
        # Initialise other attributes
        self.maxRedshift = None
        self.fieldSize = None
        return


    def setFieldOfView(self,fieldSize,footprint="square"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.fieldSize = np.radians(fieldSize)
        FOV = ET.SubElement(self.root,"fieldOfView")
        ET.SubElement(FOV,"geometry").text = footprint
        ET.SubElement(FOV,"length").text = str(fieldSize)
        return
    
    def setMaxDistance(self,maxRedshift,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.maxRedshift is None:
            self.maxRedshift = maxRedshift
        self.maxDistance = self.simulation.cosmology.comoving_distance(self.maxRedshift)
        if verbose:
            print(funcname+"(): Maximum redshift = "+str(self.maxRedshift))
            print(funcname+"(): Maximum distance for lightcone = "+str(self.maxDistance)+\
                      " "+self.simulation.boxSizeUnits)
        ET.SubElement(self.root,"maximumDistance").text = str(self.maxDistance)
        return

    def setOrigin(self,X=None,Y=None,Z=None,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        ORIGIN = ET.SubElement(self.root,"origin")
        if X is None:
            X = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if X < 0.0 or X > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): X is outside the simulation box!")
        ET.SubElement(ORIGIN,"coordinate").text = str(X)
        report = "     X = "+str(X) + " " +self.simulation.boxSizeUnits
        if Y is None:
            Y = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if Y < 0.0 or Y > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): Y is outside the simulation box!")
        ET.SubElement(ORIGIN,"coordinate").text = str(Y)
        report = report+"\n     Y = "+str(Y) + " " +self.simulation.boxSizeUnits
        if Z is None:
            Z = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if Z < 0.0 or Z > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): Z is outside the simulation box!")
        ET.SubElement(ORIGIN,"coordinate").text = str(Z)
        report = report+"\n     Z = "+str(Z) + " " +self.simulation.boxSizeUnits
        if verbose:
            print(funcname+"(): Origin for lightcone:\n"+report)
        return

    def output(self,filePath,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        tree = ET.ElementTree(self.root)
        tree.write(filePath)
        formatParametersFile(filePath)
        print(funcname+"(): geometry file written to "+filePath)
        return



        
