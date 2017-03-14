#! /usr/bin/env python

import sys,os,fnmatch
import numpy as np
from ..io import GalacticusHDF5
from ..xmlTree import xmlTree
from ..hdf5 import HDF5
from .utils import getRaDec
from ..utils.progress import Progress


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





class writeGalacticusLightcone(HDF5):
    
    def __init__(self,fileName,format="galacticus",NSIDE=None,nest=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Initalise HDF5 class
        super(writeGalacticusLightcone, self).__init__(fileName,"a")
        # Store HealPix parameters (if specified)        
        self.NSIDE = NSIDE
        self.nest = nest
        # Store format for file        
        if format.lower() not in ["galacticus","lightcone"]:
            raise ValueError(classname+"(): format must be 'galacticus' or 'lightcone'!")
        self.format = format.lower()
        # Create outputs directory
        self.mkGroup("Outputs")
        if fnmatch.fnmatch(self.format,"lightcone"):
            self.mkGroup("Outputs/nodeData")
        return


    def buildOutputsDirectory(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Make group to store outputs
        self.mkGroup("Outputs")
        outputNames = self.selectOutputsToProcess(galHDF5Obj)
        for outputName in outputNames:
            self.mkGroup("Outputs/"+outputName)
            attrib = galHDF5Obj.readAttributes("Outputs/"+outputName)
            self.addAttributes("Outputs/"+outputName,attrib)
        return


    


    def createOutputDirectory(self,galHDF5Obj,outputName):        
        if outputName not in self.fileObj["Outputs"].keys():
            self.mkGroup("Outputs/"+outputName)
            attrib = galHDF5Obj.readAttributes("Outputs/"+outputName)
            self.addAttributes("Outputs/"+outputName,attrib)
            self.mkGroup("Outputs/"+outputName+"/nodeData")
        return



    def getRedshiftMask(self,z,redshiftRange=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        mask = np.ones(len(z),dtype=bool)
        if redshiftRange is not None:
            mask = np.logical_and(z>=redshiftRange[0],z<=redshiftRange[1])
        return mask


    def getPixelMask(self,X,Y,Z,pixelNumber=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        mask = np.ones(len(X),dtype=bool)
        if pixelNumber is not None:
            if self.NSIDE is None:
                raise ValueError(funcname+"(): Class value for NSIDE was not specified!")            
            from .pixels import pixelNumberValid,getPixelNumbers
            if not pixelNumberValid(self.NSIDE,pixelNumber):
                raise ValueError(funcname+"(): pixel number outside range of allowed pixels for NSIDE = "+str(NSIDE))            
            ra,dec = getRaDec(X,Y,Z,degrees=True)
            mask = getPixelNumbers(self.NSIDE,ra,dec,nest=self.nest)==pixelNumber
        return mask
            

    def addGalaxyDataset(self,datasetName,zGalHDF5Obj,path,mask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if not path.endswith("/"):
            path = path + "/"
        data = np.array(zGalHDF5Obj["nodeData/"+datasetName])
        if mask is not None:
            data = data[mask]
        if datasetName not in self.lsDatasets(path):
            attrib = dict(zGalHDF5Obj["nodeData/"+datasetName].attrs)
            append = False
        else:
            attrib = None
            append = True
        self.addDataset(path,datasetName,data,append=append,overwrite=False,\
                        maxshape=tuple([None]),chunks=True,compression="gzip",\
                        compression_opts=6)
        if attrib is not None:
            self.addAttributes(path+datasetName,attrib)
        return
            

    def maskMergerTrees(self,mask,index,count,weight,startIndex):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        dataIndex = np.copy(np.repeat(index,count))        
        uniq,uniqCount = np.unique(dataIndex[mask],return_counts=True)
        indexMask = np.array([ind in uniq for ind in index])
        newIndex = index[indexMask]
        newCount = np.array([uniqCount[np.argwhere(uniq==ind)[0][0]] for ind in newIndex])
        newWeight = np.array([weight[np.argwhere(index==ind)[0][0]] for ind in newIndex])
        newStartIndex = np.array([startIndex[np.argwhere(index==ind)[0][0]] for ind in newIndex])
        return newIndex,newCount,newWeight,newStartIndex

    def addMergerTreeFromOutput(self,galHDF5Obj,outputName,mask=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check output contains galaxies
        if "nodeData" not in galHDF5Obj.lsGroups("Outputs/"+outputName):            
            return
        if len(galHDF5Obj.lsDatasets("Outputs/"+outputName+"/nodeData"))==0:
            return
        # Create directories if necessary
        if fnmatch.fnmatch(self.format,"galacticus"):
            self.createOutputDirectory(galHDF5Obj,outputName)
            path = "Outputs/"+outputName+"/"
        else:
            if "nodeData" not in self.fileObj["Outputs"].keys():
                self.mkGroup("Outputs/nodeData")
            path = "Outputs/"
        # Extract merger tree information
        OUT = galHDF5Obj.fileObj['Outputs/'+outputName]
        count = np.array(OUT["mergerTreeCount"])
        weight = np.array(OUT["mergerTreeWeight"])
        index = np.array(OUT["mergerTreeIndex"])
        startIndex = np.array(OUT["mergerTreeStartIndex"])
        # Apply mask if necessary
        if mask is not None:
            if not all(mask):
                index,count,weight,startIndex = self.maskMergerTrees(mask,index,count,weight,startIndex)
        # Write to file
        append = False
        if "mergerTreeIndex" in self.lsDatasets(path):
            append = True
        self.addDataset(path,"mergerTreeCount",count,append=append,overwrite=False,\
                            maxshape=tuple([None]),chunks=True,compression="gzip",\
                            compression_opts=6)
        self.addDataset(path,"mergerTreeIndex",index,append=append,overwrite=False,\
                            maxshape=tuple([None]),chunks=True,compression="gzip",\
                            compression_opts=6)
        self.addDataset(path,"mergerTreeStartIndex",startIndex,append=append,overwrite=False,\
                            maxshape=tuple([None]),chunks=True,compression="gzip",\
                            compression_opts=6)
        self.addDataset(path,"mergerTreeWeight",weight,append=append,overwrite=False,\
                            maxshape=tuple([None]),chunks=True,compression="gzip",\
                            compression_opts=6)
        return

    
    def addGalaxiesFromOutput(self,galHDF5Obj,outputName,props=None,pixelNumber=None,redshiftRange=None,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check output contains galaxies
        if "nodeData" not in galHDF5Obj.lsGroups("Outputs/"+outputName):            
            return
        if len(galHDF5Obj.lsDatasets("Outputs/"+outputName+"/nodeData"))==0:
            return
        # Create directories if necessary
        if fnmatch.fnmatch(self.format,"galacticus"):
            self.createOutputDirectory(galHDF5Obj,outputName)
            path = "Outputs/"+outputName+"/nodeData/"
        else:
            if "nodeData" not in self.fileObj["Outputs"].keys():
                self.mkGroup("Outputs/nodeData")
            path = "Outputs/nodeData/"
        # Construct mask to select galaxies
        z = np.array(galHDF5Obj.fileObj["Outputs/"+outputName+"/nodeData/lightconeRedshift"])
        zmask = self.getRedshiftMask(z,redshiftRange=redshiftRange)
        if pixelNumber is not None:
            X = np.array(galHDF5Obj.fileObj["Outputs/"+outputName+"/nodeData/lightconePositionX"])
            Y = np.array(galHDF5Obj.fileObj["Outputs/"+outputName+"/nodeData/lightconePositionY"])
            Z = np.array(galHDF5Obj.fileObj["Outputs/"+outputName+"/nodeData/lightconePositionZ"])
            pmask = self.getPixelMask(X,Y,Z,pixelNumber=pixelNumber)
        else:
            pmask = np.ones(len(zmask),bool=True)
        mask = np.logical_and(zmask,pmask)
        if any(mask):
            # Update merger tree information
            if fnmatch.fnmatch(self.format,"galacticus"):
                self.addMergerTreeFromOutput(galHDF5Obj,outputName,mask=mask)
            # Select properties to add
            if props is None:
                props = galHDF5Obj.lsDatasets("Outputs/"+outputName+"/nodeData")
            # Add galaxy properties
            OUT = galHDF5Obj.fileObj["Outputs/"+outputName]
            dummy = [self.addGalaxyDataset(datasetName,OUT,path,mask=mask) for datasetName in props]
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        return


    def addGalaxies(self,galHDF5Obj,props=None,pixelNumber=None,redshiftRange=None,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Skip if file does not contain any galaxies
        if galHDF5Obj.outputs is None:
            return
        # Copy all directories apart from galaxy outputs                                                                                                                                                                
        self.addBuildInformation(galHDF5Obj)
        self.addVersionInformation(galHDF5Obj)
        self.addParametersInformation(galHDF5Obj)
        # Loop over outputs adding galaxies where necessary        
        if progressObj is None:
            PROG = Progress(len(galHDF5Obj.outputs.name))
        else:
            PROG = None
        dummy = [self.addGalaxiesFromOutput(galHDF5Obj,outName,props=props,pixelNumber=pixelNumber,redshiftRange=redshiftRange,progressObj=PROG)\
                     for outName in galHDF5Obj.outputs.name]
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        return


    def addBuildInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if "Build" not in self.lsGroups("/"):
            self.cpGroup(str(galHDF5Obj.fileObj.filename),"Build")
        return

    def addVersionInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "Version" not in self.lsGroups("/"):
            self.cpGroup(galHDF5Obj.fileObj.filename,"Version")
        return

    def addParametersInformation(self,galHDF5Obj):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if "Parameters" not in self.lsGroups("/"):
            self.cpGroup(galHDF5Obj.fileObj.filename,"Parameters")
        return




