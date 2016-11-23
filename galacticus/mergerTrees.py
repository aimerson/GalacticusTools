#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from .hdf5 import HDF5
from .cosmology import Cosmology


class GalacticusMergerTree(HDF5):
    
    def __init__(self,*args,**kwargs):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name

        # Initalise HDF5 class
        super(GalacticusMergerTree, self).__init__(*args,**kwargs)

        # Store cosmology
        COS = self.readAttributes("cosmology")
        self.cosmology = Cosmology(omega0=float(COS["OmegaMatter"]),lambda0=float(COS["OmegaLambda"]),\
                                       omegab=float(COS["OmegaBaryon"]),h0=float(COS["HubbleParam"]),\
                                       sigma8=float(COS["sigma_8"]),ns=float(COS["powerSpectrumIndex"]))

        # Store simulation data
        self.simulation = self.readAttributes("simulation")        
        self.boxSize = COS["boxSize"]
        # Store group finder information
        self.groupFinder = self.readAttributes("groupFinder")        
        # Store units information
        self.units = self.readAttributes("units")        
        
        # Store tree indices and sizes
        self._indexDir = fnmatch.filter(self.fileObj.keys(),"*Index")[0]
        totalTrees = len(np.array(self.fileObj[self._indexDir+"/firstNode"]))
        self.trees = np.zeros(totalTrees,dtype=[("firstNode",int),("numberOfNodes",int),("treeIndex",int)])
        self.trees["firstNode"] = np.array(self.fileObj[self._indexDir+"/firstNode"])
        self.trees["numberOfNodes"] = np.array(self.fileObj[self._indexDir+"/numberOfNodes"])
        self.trees["treeIndex"] = np.array(self.fileObj[self._indexDir+"/"+self._indexDir])
        self.trees = self.trees.view(np.recarray)

        # Store path to tree data
        self._treeDir = fnmatch.filter(self.fileObj.keys(),"*Trees")[0]
        self.treeProperties = list(map(str,self.fileObj[self._treeDir].keys()))
        
        return


    def getNodeIndices(self,treeIndex,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        itree = np.argwhere(self.trees.treeIndex==treeIndex)
        firstNode = self.trees.firstNode[itree][0][0]
        numberNodes = self.trees.numberOfNodes[itree][0][0]
        if verbose:
            print(funcname+"(): tree index, first node, number of nodes = "+\
                      "("+str(treeIndex)+", "+str(firstNode)+", "+str(numberNodes)+")")
        nodes = np.arange(firstNode,firstNode+numberNodes,dtype=int)
        return nodes


    def extractSingleTree(self,treeIndex,hdf5Obj,verbose=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check tree exists
        if treeIndex not in self.trees.treeIndex:
            raise ValueError(funcname+"(): tree "+str(treeIndex)+" not found!")    
        # Create mask to select halo properties        
        nodes = self.getNodeIndices(treeIndex)
        # Extract directory for tree data
        TREE = self.fileObj[self._treeDir]
        # Begin extracting properties
        higherDimensions = ["position","velocity"]
        oneDimensionalDatasets = list(set(self.treeProperties).difference(higherDimensions))
        dummy = [hdf5Obj.addDataset(self._treeDir,name,np.array(TREE[name])[nodes],append=True)\
                     for name in oneDimensionalDatasets]
        del dummy
        # Extract positions and velocities        
        dummy = [hdf5Obj.addDataset(self._treeDir,name,np.array(TREE[name])[nodes,:],append=True,maxshape=tuple([None,3]))\
                     for name in higherDimensions]
        del dummy
        return

    
    def extractTrees(self,outFile,treeIndices):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Open output file
        OUT = HDF5(outFile,'w')
        # Copy cosmology/simulation/group-finder/units information
        cpDirs = list(set(list(map(str,self.fileObj.keys()))).difference([self._indexDir,self._treeDir]))
        dummy = [OUT.cpGroup(self.filename,dir) for dir in cpDirs]
        del dummy
        # Write tree indices and node locations
        treeIndices = np.sort(treeIndices)               
        OUT.addDataset("treeIndex","treeIndex",treeIndices)
        numberNode = np.array([self.trees.numberOfNodes[np.argwhere(self.trees.treeIndex==index)[0][0]] \
                                  for index in treeIndices])
        OUT.addDataset("treeIndex","numberOfNodes",numberNode)
        firstNode = np.cumsum(numberNode) - numberNode
        OUT.addDataset("treeIndex","firstNode",firstNode)
        # Extract tree properties
        dummy = [self.extractSingleTree(treeID,OUT) for treeID in treeIndices]    
        del dummy
        # Close file and finish
        OUT.fileObj.close()
        return

