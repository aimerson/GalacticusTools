#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from .xmlTree import xmlTree

######################################################################################
# PARAMETERS
######################################################################################


class GalacticusParameters(xmlTree):
    
    def __init__(self,xmlfile,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusParameters,self).__init__(xmlfile=xmlfile,verbose=verbose)
        return
    
    def getParameter(self,param):
        """
        get_parameter: Return value of specified parameter.
        
        USAGE: value = getParameter(param)

            INPUT
                param -- name of parameter
            OUTPUT
                value -- string with value for parameter

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if not param in self.treeMap.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            value = None
        else:
            path = self.treeMap[param]
            elem = self.getElement(path)
            value = elem.attrib.get("value")
        return value


    def getParent(self,param):
        """
        getParent: Return name of parent element.

        USAGE: name = getParent(elem)

            INPUT
                elem -- name of current element
            OUTPUT
                name -- string with name of parent

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not param in self.treeMap.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            name = None
        else:
            name = self.treeMap[param]
            name = name.split("/")[-2]
        return name


    def constructDictionary(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        params = {}
        for e in self.treeMap.keys():
            params[e] = self.getParameter(e)
        return params

    
    def setParameter(self,param,value,parent=None,selfCreate=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Convert paramter value to string
        if np.ndim(value) == 0:
            value = str(value)
        else:
            value = " ".join(map(str,value))
        # Set parameter
        self.setElement(param,attrib={"value":value},parent=parent,\
                            selfCreate=selfCreate)
        return

####################################################################            

def validate_parameters(xmlfile,GALACTICUS_ROOT):
    script = GALACTICUS_ROOT+"/scripts/aux/validateParameters.pl"
    os.system(script+" "+xmlfile)
    return


def formatParametersFile(ifile,ofile=None):    
    import shutil
    tmpfile = ifile.replace(".xml","_copy.xml")
    if ofile is not None:
        cmd = "xmllint --format "+ifile+" > "+ofile
    else:
        cmd = "xmllint --format "+ifile+" > "+tmpfile
    os.system(cmd)
    if ofile is None:
        shutil.move(tmpfile,ifile)        
    return


####################################################################


def simulationParameters(simulation):
    
    params = []
    if simulation.lower() in ["millennium","milli-millennium"]:
        params.append(("treeNodeMethodSatellite","preset"))
        params.append(("treeNodeMethodPosition","preset"))
        params.append(("mergerTreeConstructMethod","read"))
        params.append(("allTreesExistAtFinalTime","false"))
        params.append(("cosmologyParametersMethod","simple"))
        params.append(("HubbleConstant",73.0,"cosmologyParametersMethod"))
        params.append(("OmegaMatter",0.25,"cosmologyParametersMethod"))
        params.append(("OmegaDarkEnergy",0.75,"cosmologyParametersMethod"))
        params.append(("OmegaBaryon",0.0455,"cosmologyParametersMethod"))
        params.append(("sigma_8",0.9,"cosmologicalMassVarianceMethod"))
        params.append(("powerSpectrumPrimordialMethod","powerLaw"))
        params.append(("index",0.961,"powerSpectrumPrimordialMethod"))
        params.append(("wavenumberReference",1,"powerSpectrumPrimordialMethod"))
        params.append(("running",0,"powerSpectrumPrimordialMethod"))
        params.append(("mergerTreeReadPresetScaleRadiiMinimumMass",2.5e11))
        params.append(("virialDensityContrastMethod","percolation"))
        params.append(("virialDensityContrastPercolationLinkingLength","0.2"))
    return params
