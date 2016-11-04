#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from galacticus.io import GalacticusHDF5
from galacticus.xmlTree import xmlTree
from galacticus.hdf5 import HDF5


class LightconeHDF5(HDF5):
    
    def __init__(self,*args,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(LightconeHDF5, self).__init__(*args,**kwargs)        
        return

    def addGalacticusInformation(self,galacticusFile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name     
        self.cpGroup(galacticusFile,"Build",dstdir="GalacticusBuild")
        self.cpGroup(galacticusFile,"Parameters",dstdir="GalacticusParameters")
        self.cpGroup(galacticusFile,"Version",dstdir="GalacticusVersion")
        return

    def addGeometryParameters(self,geometryFile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name     
        GEO = xmlTree(xmlfile=geometryFile)
        self.mkGroup("LightconeGeometry")
        self.addAttributes("LightconeGeometry",{"boxLength":GEO.getElement("geometry/boxLength").text})
        self.mkGroup("LightconeGeometry/fieldOfView")
        self.addAttributes("LightconeGeometry/fieldOfView",{"geometry":GEO.getElement("geometry/fieldOfView/geometry").text})
        self.addAttributes("LightconeGeometry/fieldOfView",{"length":GEO.getElement("geometry/fieldOfView/length").text})
        ORIGIN = GEO.getElement("geometry/origin")
        coordinates = []
        for coord in ORIGIN.findall("coordinate"):
            coordinates.append(coord.text)
        self.addAttributes("LightconeGeometry",{"origin":coordinates})
        self.mkGroup("LightconeGeometry/unitVectors")
        UNIT = GEO.getElement("geometry/unitVector1")
        coordinates = []
        for coord in UNIT.findall("coordinate"):
            coordinates.append(coord.text)
        self.addAttributes("LightconeGeometry/unitVectors",{"unitVector1":coordinates})
        UNIT = GEO.getElement("geometry/unitVector2")
        coordinates = []
        for coord in UNIT.findall("coordinate"):
            coordinates.append(coord.text)
        self.addAttributes("LightconeGeometry/unitVectors",{"unitVector2":coordinates})
        UNIT = GEO.getElement("geometry/unitVector3")
        coordinates = []
        for coord in UNIT.findall("coordinate"):
            coordinates.append(coord.text)
        self.addAttributes("LightconeGeometry/unitVectors",{"unitVector3":coordinates})
        self.addAttributes("LightconeGeometry",{"maximumDistance":GEO.getElement("geometry/maximumDistance").text})
        self.mkGroup("LightconeGeometry/outputs")
        OUT = GEO.getElement("geometry/outputs")
        for name in "minimumDistance maximumDistance redshift".split():
            output = []
            for coord in OUT.findall(name):            
                output.append(coord.text)
            self.addDataset("LightconeGeometry/outputs",name,np.copy(np.array(output).astype(float)))
        return
            

    def addGalaxyProperty(self,datasetName,data):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name     
        newDatasetName = datasetName.split(":")
        redshift = fnmatch.filter(newDatasetName,"z*")
        if len(redshift) > 0:
            newDatasetName = datasetName.replace(":"+redshift[0],"")
        else:
            newDatasetName = datasetName
        self.addDataset("Output",newDatasetName,data,append=True)
        return

    def addGalaxies(self,galaxies):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name     
        dummy = [self.addGalaxyProperty(name,galaxies[name]) for name in galaxies.dtype.names]
        del dummy
        return


