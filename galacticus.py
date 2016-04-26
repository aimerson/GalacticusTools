#! /usr/bin/env python

import numpy as np
import xml.etree.ElementTree as ET


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
        if param in self.parent_map.keys():        
            # Update existing parameter
            elem = self.parent_map[param]
            par = elem.find(param)
            par.set("value",str(value))
        else:
            # Insert new parameter
            if parent is None or parent == "parameters":
                parent_elem = self.params
                parent_set = True
            else:
                parent_elem = self.create_path(parent,delim=delim)
            ET.SubElement(parent_elem,param,attrib={"value":str(value)})
            self.parent_map[param] = parent_elem
        return

    def write(self,ofile):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.tree.write(ofile)





GAL = GalacticusParameters("parameters_ref.xml",verbose=True)
print GAL.get_parameter("sigma_8")
parent = GAL.get_parent("sigma_8")
GAL.set_parameter("sigma_10",1.0,parent="parameters/parameters2")
print GAL.get_parameter("sigma_10")
GAL.write("test.xml")
print "done"
quit()



