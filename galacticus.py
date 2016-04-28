#! /usr/bin/env python

import os,sys
import numpy as np
import xml.etree.ElementTree as ET
import h5py


######################################################################################
# PARAMETERS
######################################################################################

class GalacticusParameters(object):
    
    def __init__(self,xmlfile,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.xmlfile = xmlfile
        self.tree = ET.parse(self.xmlfile)
        self.params = self.tree.getroot()
        self._verbose = verbose
        self.parent_map = {c.tag:p for p in self.params.iter() for c in p}
        return
    
    def get_parameter(self,param):        
        """
        get_parameter: Return value of specified parameter.
        
        USAGE: value = get_parameter(param)

            INPUT
                param -- name of parameter
            OUTPUT
                value -- string with value for parameter

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if not param in self.parent_map.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            value = None
        else:
            elem = self.parent_map[param]
            value = elem.find(param).attrib.get("value")
        return value

    def get_parent(self,param):
        """
        get_parent: Return name of parent element.

        USAGE: name = get_parent(elem)

            INPUT
                elem -- name of current element
            OUTPUT
                name -- string with name of parent

        """
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not param in self.parent_map.keys():
            if self._verbose:
                print("WARNING! "+funcname+"(): Parameter '"+\
                          param+"' cannot be located in "+self.xmlfile)
            name = None
        else:
            name = self.parent_map[param].tag
        return name

    def get_dict(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        params = {}
        for e in self.params.iter():
            params[e.tag] = e.attrib.get("value")
        return params
    
    def create_path(self,new_path,delim="/"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        path = new_path.split(delim)
        if path[0] == "parameters":
            dummy = path.pop(0)
        parent_elem = self.params
        for node in path:
            if node in self.parent_map.keys(): 
                if parent_elem.find(node) is None:
                    parent_elem = ET.SubElement(parent_elem,node)            
                else:
                    parent_elem = parent_elem.find(node)
            else:
                parent_elem = ET.SubElement(parent_elem,node)            
        self.parent_map = {c.tag:p for p in self.params.iter() for c in p}        
        return parent_elem

    def set_parameter(self,param,value,parent=None,delim="/"):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Convert paramter value to string
        if np.ndim(value) == 0:
            value = str(value)
        else:
            value = " ".join(map(str,value))
        if param in self.parent_map.keys():        
            # Update existing parameter
            elem = self.parent_map[param]
            par = elem.find(param)
            par.set("value",value)
        else:
            # Insert new parameter
            if parent is None or parent == "parameters":
                parent_elem = self.params
                parent_set = True
            else:
                parent_elem = self.create_path(parent,delim=delim)
            ET.SubElement(parent_elem,param,attrib={"value":value})
            self.parent_map[param] = parent_elem
        return

    def write(self,ofile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.tree.write(ofile)


def validate_parameters(xmlfile,GALACTICUS_ROOT):
    script = GALACTICUS_ROOT+"/scripts/aux/validateParameters.pl"
    os.system(script+" "+xmlfile)
    return


######################################################################################
# FILE I/O
######################################################################################


def check_readonly(func):
    def wrapper(self,*args,**kwargs):
        funcname = self.__class__.__name__+"."+func.__name__
        if self.read_only:
            raise IOError(funcname+"(): HDF5 file "+self.filename+" is READ ONLY!")
        return func(self,*args,**kwargs)
    return wrapper


class GalacticusHDF5(object):
    
    def __init__(self,*args,**kwargs):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "verbose" in kwargs.keys():
            self._verbose = kwargs["verbose"]
        else:
            self._verbose = False

        # Open file and store file object
        self.fileObj = h5py.File(*args)

        # Store file name and access mode
        self.filename = self.fileObj.filename
        if self._verbose:
            print(classname+"(): HDF5 file = "+self.filename)
        if self.fileObj.mode == "r":
            self.read_only = True
            if self._verbose:
                print(classname+"(): HDF5 opened in READ-ONLY mode")
        elif self.fileObj.mode == "r+":
            self.read_only = False

        # Store version information
        self.version = dict(self.fileObj["Version"].attrs)
        
        # Store build information
        self.build = dict(self.fileObj["Build"].attrs)

        # Store parameters
        self.parameters = dict(self.fileObj["Parameters"].attrs)
        self.parameters_parents = { k:"parameters" for k in self.fileObj["Parameters"].attrs.keys()}
        for k in self.fileObj["Parameters"]:
            if len(self.fileObj["Parameters/"+k].attrs.keys())>0:
                d = dict(self.fileObj["Parameters/"+k].attrs)                
                self.parameters.update(d)
                d = { a:k for a in self.fileObj["Parameters/"+k].attrs.keys()}
                self.parameters_parents.update(d)

        # Store output epochs
        Outputs = self.fileObj["Outputs"]
        nout = len(Outputs.keys())
        self.outputs = np.zeros(nout,dtype=[("iout",int),("a",float),("z",float)])
        for i,out in enumerate(Outputs.keys()):
            self.outputs["iout"][i] = int(out.replace("\n","").replace("Output",""))
            a = float(Outputs[out].attrs["outputExpansionFactor"])
            self.outputs["a"][i] = a
            self.outputs["z"][i] = (1.0/a) - 1.0
        self.outputs = self.outputs.view(np.recarray)

        return


    def close(self):
        self.fileObj.close()
        return


    def global_history(self,props=None,si=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        globalHistory = self.fileObj["globalHistory"]
        allprops = globalHistory.keys() + ["historyRedshift"]
        if props is None:
            props = allprops 
        else:
            props = set(props).intersection(allprops)            
        epochs = len(np.array(globalHistory["historyExpansion"]))
        dtype = np.dtype([ (str(p),np.float) for p in props ])
        history = np.zeros(epochs,dtype=dtype)
        if self._verbose:
            if si:
                print("WARNING! "+funcname+"(): Adopting SI units!")
            else:
                print("WARNING! "+funcname+"(): NOT adopting SI units!")        
        for p in history.dtype.names:
            if p is "historyRedshift":
                history[p] = np.copy((1.0/np.array(globalHistory["historyExpansion"]))-1.0)
            else:
                history[p] = np.copy(np.array(globalHistory[p]))
                if si:
                    if "unitsInSI" in globalHistory[p].attrs.keys():
                        unit = globalHistory[p].attrs["unitsInSI"]
                        history[p] = history[p]*unit
        return history.view(np.recarray)


    def galaxies(self,props=None,z=None):
        iselect = np.argmin(np.fabs(outputs.z-kwargs["z"]))
        out = Outputs["Output"+str(outputs["iout"][iselect])]
        return




class GALACTICUS(object):
    def __init__(self):
        return


