#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
import xml.etree.ElementTree as ET
from .xmlTree import xmlTree

######################################################################################
# PARAMETERS
######################################################################################


class GalacticusParameters(xmlTree):
    
    def __init__(self,xmlfile,root='parameters',verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(GalacticusParameters,self).__init__(xmlfile=xmlfile,root=root,verbose=verbose)
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
# FILTERS CLASS
####################################################################            

class FiltersParameters(object):

    def __init__(self,defaultAbsorptionMethods=["inoue2014"],recentAbsorptionMethods=["inoue2014"],recentTimeLimit=1.0e-2,\
                     quitOnError=True):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create lists to store filter information
        self.luminosityFilter = []
        self.luminosityType = []
        self.luminosityRedshift = []
        self.luminosityPostProcessSet = []
        # Set lists of allowed values
        self.allowedPostProcessSets = ["default","recent","unabsorbed","recentUnabsorbed"]
        self.allowedAbsorptionMethods = ["inoue2014","meiksin2006","madau1995","lycSuppress","identity"]
        # Set list of methods for default post-processing
        goodMethods = list(set(defaultAbsorptionMethods).intersection(self.allowedAbsorptionMethods))
        badMethods = list(set(defaultAbsorptionMethods).difference(self.allowedAbsorptionMethods))
        if len(badMethods)>0:
            if quitOnError:
                report = classname+"(): defaultMethods: some methods not recognised!" +\
                    "\n     Not recognised  = "+" ,".join(badMethods)+\
                    "\n     Allowed methods = "+" ,".join(self.allowedAbsorptionMethods)
                raise ValueError(report)
        self.defaultMethods = goodMethods
        # Set list of methods for recent star formation
        goodMethods = list(set(recentAbsorptionMethods).intersection(self.allowedAbsorptionMethods))
        badMethods = list(set(recentAbsorptionMethods).difference(self.allowedAbsorptionMethods))
        if len(badMethods)>0:
            if quitOnError:
                report = classname+"(): recentMethods: some methods not recognised!" +\
                    "\n     Not recognised  = "+" ,".join(badMethods)+\
                    "\n     Allowed methods = "+" ,".join(self.allowedAbsorptionMethods)
                raise ValueError(report)
        self.recentMethods = goodMethods + ["recent"]
        # Set methods for unabsorbed cases
        self.unabsorbedMethods = ["identity"]
        self.recentUnabsorbedMethods = ["recent"]
        return

    def addFilter(self,name,filterType=["rest","observed"],postProcess=["default"],redshift=["all"]):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Ensure options are list
        if type(filterType) is not list:
            filterType = [filterType]
        if type(postProcess) is not list:
            postProcess = [postProcess]
        if type(redshift) is not list:
            redshift = [redshift]
        # Append filter to stored list
        for luminosityType in filterType:
            for luminosityPostProcess in postProcess:
                for luminosityRedshift in redshift:
                    self.luminosityFilter.append(name)
                    self.luminosityType.append(luminosityType)
                    self.luminosityRedshift.append(luminosityRedshift)
                    self.luminosityPostProcessSet.append(luminosityPostProcess)
        return

    def addToGalacticusParameters(self,galacticusParametersObj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if len(self.luminosityFilter) == 0:
            return
        # Add filter set to parameters tree
        galacticusParametersObj.setElement("luminosityFilter",attrib={"value":" ".join(self.luminosityFilter)},selfCreate=True)
        galacticusParametersObj.setElement("luminosityType",attrib={"value":" ".join(self.luminosityType)},selfCreate=True)
        galacticusParametersObj.setElement("luminosityRedshift",attrib={"value":" ".join(self.luminosityRedshift)},selfCreate=True)
        galacticusParametersObj.setElement("luminosityPostprocessSet",attrib={"value":" ".join(self.luminosityPostProcessSet)},selfCreate=True)
        # Add methods to parameters tree
        galacticusParametersObj.setElement("stellarPopulationSpectraPostprocessDefaultMethods",\
                                               attrib={"value":" ".join(self.defaultMethods)},selfCreate=True)
        if "recent" in list(np.unique(self.luminosityPostProcessSet)):
            galacticusParametersObj.setElement("stellarPopulationSpectraPostprocessRecentMethods",\
                                                   attrib={"value":" ".join(self.recentMethods)},selfCreate=True)
        if "unabsorbed" in list(np.unique(self.luminosityPostProcessSet)):
            galacticusParametersObj.setElement("stellarPopulationSpectraPostprocessUnabsorbedMethods",\
                                                   attrib={"value":" ".join(self.unabsorbedMethods)},selfCreate=True)
        if "recentUnabsorbed" in list(np.unique(self.luminosityPostProcessSet)):
            galacticusParametersObj.setElement("stellarPopulationSpectraPostprocessRecentUnabsorbedMethods",\
                                                   attrib={"value":" ".join(self.recentUnabsorbedMethods)},selfCreate=True)
        return


####################################################################            
# MISC FUNCTIONS
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

