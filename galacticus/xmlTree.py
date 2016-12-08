#! /usr/bin/env python

import os,sys
import numpy as np
import xml.etree.ElementTree as ET

def formatFile(ifile,ofile=None):
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


def mapElements(root):
    elementMap = {}
    for p in root.iter():
        parent = p.tag
        if parent not in elementMap.keys():
            elementMap[parent] = p.tag
        parentPath = elementMap[parent]
        for c in p:
            child = c.tag
            path = parentPath + "/" + c.tag
            elementMap[c.tag] = path
    return elementMap



class xmlTree(object):
    
    def __init__(self,xmlfile=None,root="root",verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self._verbose = verbose
        self.xmlfile = xmlfile         
        if self.xmlfile is None:
            root = ET.Element(root)
            self.tree = ET.ElementTree(element=root)
            self.root = self.tree.getroot()                
            self.treeMap = {}
        else:
            self.tree = ET.parse(self.xmlfile)
            self.root = self.tree.getroot()                
            self.treeMap = mapElements(self.root)
        if self._verbose:
            print(classname+"(): Root is '"+self.root.tag+"'")            
        return
        
    def getElement(self,path):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        nodes = path.split("/")     
        if len(nodes) == 1:
            elem = self.root
        else:
            node = nodes.pop()
            elem = self.getElement("/".join(nodes)).find(node)                        
        return elem

    def updateTreeMap(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.treeMap = mapElements(self.root)
        return

    def createElement(self,name,attrib={},parent=None,text=None,overwrite=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if name in self.treeMap.keys() and not overwrite:
            return
        if name in self.treeMap.keys() and overwrite:
            elem = getElement(self.treeMap[name]).clear
        if parent is None or parent==self.root.tag:
            path = self.root.tag
            parent = self.root
        else:
            path = self.treeMap[parent]
            parent = self.getElement(path)
        if parent is None:
            raise ValueError(funcname+"(): error in path to parent element -- some nodes missing?"+\
                                 "    \nParent path = "+path)        
        if text is None:
            ET.SubElement(parent,name,attrib=attrib)
        else:
            ET.SubElement(parent,name,attrib=attrib).text = str(text)
        self.treeMap[name] = path+"/"+name
        return
    
    def setElement(self,name,attrib={},text=None,parent=None,selfCreate=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if name not in self.treeMap.keys():
            if selfCreate:
                self.createElement(name,attrib=attrib,parent=parent,text=text)
            else:
                raise KeyError(funcname+"(): element '"+name+"' does not exist!")
        elem = self.getElement(self.treeMap[name])
        dummy = [elem.set(k,attrib[k]) for k in attrib.keys()]
        del dummy
        if text is not None:
            elem.text = text
        return

    def appendElement(self,newBranch,parent=None,updateMap=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if parent is None:
            node = self.root
        elif type(parent) == str:
            if parent.endswith("/"):
                parent = parent[:-1]
            nodes = parent.split("/")
            if nodes[0] == self.root.tag:
                nodes = nodes[1:]
            parentNode = self.root
            for nodeName in nodes:
                node = parentNode.find(nodeName)
                if node is None:
                    self.createElement(nodeName,parent=parentNode.tag)
                else:
                    parentNode = node            
        else:
            node = parent
        if newBranch.tag in list(node):
            elem = self.getElement(newBranch.tag)
            node.remove(elem)
        node.append(newBranch)
        # Update (re-make) element map
        if updateMap:
            self.updateTreeMap()
        return
                       
    def writeToFile(self,outFile,format=True):
        self.tree.write(outFile)
        if format:
            formatFile(outFile)
        return

