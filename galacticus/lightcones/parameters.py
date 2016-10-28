#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from ..xmlTree import xmlTree
from ..parameters import GalacticusParameters



class LightconeParameters(GalacticusParameters):
    
    def __init__(self,xmlfile=None,root='lightcone',verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if xmlfile is not None:
            if not os.path.exists(xmlfile):
                xmlfile = None
        super(LightconeParameters,self).__init__(xmlfile=xmlfile,root=root,verbose=verbose)        
        return

    def addGeometryParameters(self,geometryFile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Load geometry file into XML class
        if not os.path.exists(geometryFile):
            raise  IOError(funcname+"(): cannot locate geometry file '"+geometryFile+"'!")        
        GEO = xmlTree(xmlfile=geometryFile,verbose=self._verbose)
        self.createElement("geometry")
        for name in "fieldOfView origin unitVector1 unitVector2 unitVector3".split():
            path = "geometry/" + name
            elem = GEO.getElement(path)
            self.appendElement(elem,parent="geometry")
        return

    def setPathParameters(self,galacticusDir,lightconeDir=None,\
                               galacticusPrefix="galacticus",lightconePrefix="lightcone"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if lightconeDir is None:
            lightconeDir = galacticusDir
        self.createElement("ioPaths")
        self.createElement("galacticusDir",parent="ioPaths",attrib={"value":galacticusDir})
        self.createElement("galacticusPrefix",parent="ioPaths",attrib={"value":galacticusPrefix})
        self.createElement("lightconeDir",parent="ioPaths",attrib={"value":lightconeDir})
        self.createElement("lightconePrefix",parent="ioPaths",attrib={"value":lightconePrefix})
        return

    


        
        
    
    
