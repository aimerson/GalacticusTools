#! /usr/bin/env python

import sys
import numpy as np
import xml.etree.ElementTree as ET
from ..simulations import Simulation
from ..parameters import formatParametersFile
from ..constants import Parsec,megaParsec
from .footprint import KitzbichlerWhite2007



class Geometry(object):
    
    def __init__(self,treeRoot):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.root = treeRoot
        return




class geometryLightconeMethod(Geometry):
    
    def __init__(self,simulationName,fieldOfView,method='square',timeEvolvesAlongLightcone=True):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Intialise simulation object
        self.simulation = Simulation(simulationName)
        # Initialise XML tree
        root = ET.Element("geometryLightconeMethod",attrib={"value":method})
        self._method = method
        # Initalise Geometry class
        super(geometryLightconeMethod,self).__init__(root)
        # Set simulation specific parameters
        ET.SubElement(self.root,"lengthReplication",attrib={"value":str(self.simulation.boxSize)})
        ET.SubElement(self.root,"lengthHubbleExponent",attrib={"value":str(-1)})
        if "gpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e9
        elif "mpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e6
        elif "kpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e3
        else:
            siValue = Parsec
        ET.SubElement(self.root,"lengthUnitsInSI",attrib={"value":str(siValue)})
        # Set whether including time evolution
        if timeEvolvesAlongLightcone:                
            ET.SubElement(self.root,"timeEvolvesAlongLightcone",attrib={"value":"true"})
        else:
            ET.SubElement(self.root,"timeEvolvesAlongLightcone",attrib={"value":"false"})
        # Set field of view
        self._fieldOfView = fieldOfView
        ET.SubElement(self.root,"angularSize",attrib={"value":str(self._fieldOfView)})
        return
        
    def setOrigin(self,X=None,Y=None,Z=None,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if X is None:
            X = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if X < 0.0 or X > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): X is outside the simulation box!")
        report = "     X = "+str(X) + " " +self.simulation.boxSizeUnits
        if Y is None:
            Y = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if Y < 0.0 or Y > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): Y is outside the simulation box!")
        report = report+"\n     Y = "+str(Y) + " " +self.simulation.boxSizeUnits
        if Z is None:
            Z = np.random.rand(1)[0]*float(self.simulation.boxSize)
        else:
            if Z < 0.0 or Z > float(self.simulation.boxSize):
                raise ValueError(funcname+"(): Z is outside the simulation box!")
        report = report+"\n     Z = "+str(Z) + " " +self.simulation.boxSizeUnits
        ET.SubElement(self.root,"origin",attrib={"value":str(X)+" "+str(Y)+" "+str(Z)})
        if verbose:
            print(funcname+"(): Origin for lightcone:\n"+report)
        return

    def setUnitVectors(self,verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        KW = KitzbichlerWhite2007(self.simulation.boxSize,boxUnits=self.simulation.boxSizeUnits)
        KW.setFieldOfView(fieldSize=self._fieldOfView,footprint=self._method,verbose=verbose)            
        KW.createUnitVectors(fieldSize,verbose=verbose)
        ET.SubElement(self.root,"unitVector1",attrib={"value":" ".join([str(i) for i in KW.unitX])})
        ET.SubElement(self.root,"unitVector2",attrib={"value":" ".join([str(i) for i in KW.unitY])})
        ET.SubElement(self.root,"unitVector3",attrib={"value":" ".join([str(i) for i in KW.unitZ])})
        return        

    def setRedshifts(self,zmin=0.0,zmax=3.0,redshifts=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if redshifts is None:
            redshifts = self.simulation.snapshots.z
            mask = np.logical_and(redshifts>=zmin,redshfits<=zmax)
            redshifts = np.sort(redshifts[mask])
        ET.SubElement(self.root,"redshift",attrib={"value":" ".join([str(z) for z in redshifts])})
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
        # Write redshifts and distances
        redshifts = np.sort(self.simulation.snapshots.z)
        distance = self.simulation.cosmology.comoving_distance(redshifts)
        minDistance = np.zeros_like(redshifts)
        maxDistance = np.zeros_like(redshifts)
        for i in range(len(redshifts)):
            if i == 0:
                minDistance[i] = distance[i]
            else:
                minDistance[i] = (distance[i] + distance[i-1])/2.0
            if i == len(redshifts) - 1:
                maxDistance[i] = distance[i]
            else:
                maxDistance[i] = (distance[i] + distance[i+1])/2.0
        OUT = ET.SubElement(self.root,"outputs")        
        def addDistance(name,value):
            ET.SubElement(OUT,name).text = str(value)
        dummy = [addDistance("maximumDistance",dist) for dist in maxDistance[::-1]]
        dummy = [addDistance("minimumDistance",dist) for dist in minDistance[::-1]]
        dummy = [addDistance("redshift",z) for z in redshifts[::-1]]
        del dummy
        # Write units
        UNIT = ET.SubElement(self.root,"units")
        LEN = ET.SubElement(UNIT,"length")
        ET.SubElement(LEN,"hubbleExponent").text = str(-1)
        if "gpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e9
        elif "mpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e6
        elif "kpc" in self.simulation.boxSizeUnits.lower():
            siValue = Parsec*1.0e3
        else:
            siValue = Parsec
        ET.SubElement(LEN,"unitsInSI").text = str(siValue)
        # Initialise other attributes
        self.maxRedshift = None
        self.fieldSize = None
        return


    def setFieldOfView(self,fieldSize=None,footprint="square"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if fieldSize is None:
            raise ValueError(funcname+"(): no field size specified!")
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

    def setUnitVectors(self,fieldSize=None,footprint='square',verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        KW = KitzbichlerWhite2007(self.simulation.boxSize,boxUnits=self.simulation.boxSizeUnits)
        if self.fieldSize is None:
            KW.setFieldOfView(fieldSize=fieldSize,footprint=footprint,verbose=verbose)            
        KW.createUnitVectors(fieldSize,verbose=verbose)
        VEC = ET.SubElement(self.root,"unitVector1")
        ET.SubElement(VEC,"coordinate").text = str(KW.unitX[0])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitX[1])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitX[2])
        VEC = ET.SubElement(self.root,"unitVector2")
        ET.SubElement(VEC,"coordinate").text = str(KW.unitY[0])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitY[1])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitY[2])
        VEC = ET.SubElement(self.root,"unitVector3")
        ET.SubElement(VEC,"coordinate").text = str(KW.unitZ[0])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitZ[1])
        ET.SubElement(VEC,"coordinate").text = str(KW.unitZ[2])
        return

    def output(self,filePath='geometry.xml',verbose=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        tree = ET.ElementTree(self.root)
        tree.write(filePath)
        formatParametersFile(filePath)
        print(funcname+"(): geometry file written to "+filePath)
        return



        
