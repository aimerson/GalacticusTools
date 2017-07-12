#! /usr/bin/env python

import os,sys,fnmatch
import numpy as np
from .hdf5 import HDF5
from .utils.datatypes import getDataType
from .utils.progress import Progress
from .cosmology import Cosmology
from .lightcones.utils import getRaDec

# Special cases for dataset names
special_cases = ["weight","mergerTreeWeight","snapshotRedshift","lightconeRightAscension","lightconeDeclination"]


######################################################################################
# FILE I/O
######################################################################################

class GalacticusHDF5(HDF5):
    
    def __init__(self,*args,**kwargs):        
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        
        # Initalise HDF5 class
        super(GalacticusHDF5, self).__init__(*args,**kwargs)

        # Store version information
        self.version = dict(self.fileObj["Version"].attrs)
        
        # Store build information if available
        try:            
            self.build = dict(self.fileObj["Build"].attrs)
        except KeyError:
            self.build = None

        # Store parameters
        self.parameters = dict(self.fileObj["Parameters"].attrs)
        self.parameters_parents = { k:"parameters" for k in self.fileObj["Parameters"].attrs.keys()}
        for k in self.fileObj["Parameters"]:
            if len(self.fileObj["Parameters/"+k].attrs.keys())>0:
                d = dict(self.fileObj["Parameters/"+k].attrs)                
                self.parameters.update(d)
                d = { a:k for a in self.fileObj["Parameters/"+k].attrs.keys()}
                self.parameters_parents.update(d)

        # Store cosmology object        
        self.cosmology = Cosmology(omega0=float(self.parameters["OmegaMatter"]),\
                                       lambda0=float(self.parameters["OmegaDarkEnergy"]),\
                                       omegab=float(self.parameters["OmegaBaryon"]),\
                                       h0=float(self.parameters["HubbleConstant"])/100.0,\
                                       sigma8=float(self.parameters["sigma_8"]),\
                                       ns=float(self.parameters["index"]),\
                                       h_independent=False)
        
        # Store output epochs
        self.outputs = None
        if "Outputs" in self.fileObj.keys():
            Outputs = self.fileObj["Outputs"]
            outputKeys = fnmatch.filter(Outputs.keys(),"Output*")
            nout = len(outputKeys)
            if nout > 0:
                isort = np.argsort(np.array([ int(key.replace("Output","")) for key in outputKeys]))
                self.outputs = np.zeros(nout,dtype=[("iout",int),("a",float),("z",float),("name","|S10")])
                for i,out in enumerate(np.array(Outputs.keys())[isort]):
                    self.outputs["name"][i] = out
                    self.outputs["iout"][i] = int(out.replace("\n","").replace("Output",""))
                    a = float(Outputs[out].attrs["outputExpansionFactor"])
                    self.outputs["a"][i] = a
                    self.outputs["z"][i] = (1.0/a) - 1.0
                self.outputs = self.outputs.view(np.recarray)
        return

    def availableDatasets(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        out = self.selectOutput(z)
        if out is None:
            return []
        return map(str,out["nodeData"].keys())

    def countGalaxies(self,z=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return 0
        if z is None:
            redshifts = self.outputs.z
        else:
            redshifts = [z]
        galaxies = np.array([self.countGalaxiesAtRedshift(redshift) for redshift in redshifts])
        return np.sum(galaxies)

    def countGalaxiesAtRedshift(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        ngals = 0
        OUT = self.selectOutput(z)
        if OUT is None:
            return ngals
        if "nodeData" in OUT.keys():            
            if len(self.availableDatasets(z)) > 0:
                dataset = self.availableDatasets(z)[0]            
                ngals = len(np.array(OUT["nodeData/"+dataset]))
        return ngals

    def datasetExists(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return len(fnmatch.filter(self.availableDatasets(z),datasetName))>0

    def getRedshiftString(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        return fnmatch.filter(fnmatch.filter(self.availableDatasets(z),"*z[0-9].[0-9]*")[0].split(":"),"z*")[0]
    
    def getUUID(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        keys = list(map(str,self.fileObj["/"].attrs.keys()))
        uuid = None
        if "UUID" in keys:            
            uuid = str(self.fileObj["/"].attrs["UUID"])
        return uuid

    def globalHistory(self,props=None,si=False):        
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

    def nearestRedshift(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return None
        # Select epoch closest to specified redshift
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        return self.outputs.z[iselect]


    def readGalaxiesOLD(self,z,props=None,SIunits=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select epoch closest to specified redshift
        out = self.selectOutput(z)
        # Set list of all available properties
        allprops = self.availableDatasets(z)
        # Get number of galaxies
        ngals = len(np.array(out["nodeData/"+allprops[0]]))
        if self._verbose:
            print(funcname+"(): Number of galaxies = "+str(ngals))
        # Read all properties if not specified
        if props is None:
            props = allprops        
        # Check for properties not already calculated
        if len(list((set(props).difference(allprops))))>0:
            self.calculateProperties(list(set(props).difference(allprops)),z,overwrite=False)
        # Construct datatype for galaxy properties to read        
        dtype = []
        for p in props:
            if len(fnmatch.filter(allprops,p))>0:
                matches = fnmatch.filter(allprops,p)
                dtype = dtype + [ (str(m),getDataType(out["nodeData/"+m])) for m in matches ]                
            else:
                if p.lower() == "weight":
                    dtype.append((p.lower(),float))
        galaxies = np.zeros(ngals,dtype=dtype)
        # Extract galaxy properties
        for p in galaxies.dtype.names:
            if p in allprops:
                galaxies[p] = np.copy(np.array(out["nodeData/"+p]))
                if SIunits:
                    if "unitsInSI" in out["nodeData/"+p].attrs.keys():
                        unit = out["nodeData/"+p].attrs["unitsInSI"]
                        galaxies[p] *= unit
            else:
                if p.lower() == "weight":
                    cts = np.array(out["mergerTreeCount"])
                    wgt = np.array(out["mergerTreeWeight"])
                    galaxies[p.lower()] = np.copy(np.repeat(wgt,cts))
                    del cts,wgt            
        return galaxies



    def readGalaxies(self,z,props=None,SIunits=False,removeRedshiftString=False):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create array to store galaxy data
        self.galaxies = None
        # Read galaxies from one or more outputs
        if np.ndim(z) == 1:
            if len(z) == 1:
                z = z[0]
        if np.ndim(z) == 0:
            self.readGalaxiesAtRedshift(z,props=props,SIunits=SIunits,removeRedshiftString=removeRedshiftString)
        else:
            zout = np.unique([self.outputs.z[np.argmin(np.fabs(self.outputs.z-iz))] for iz in z])
            PROG = Progress(len(zout))
            dummy = [self.readGalaxiesAtRedshift(iz,props=props,SIunits=SIunits,removeRedshiftString=True,progressObj=PROG) for iz in zout]
        return self.galaxies

    
    def readGalaxiesAtRedshift(self,z,props=None,SIunits=False,removeRedshiftString=False,progressObj=None):                
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Initiate class for snapshot output
        OUTPUT = SnapshotOutput(z,self)
        # Read galaxies from snapshot
        OUTPUT.readGalaxies(props=props,SIunits=SIunits,removeRedshiftString=removeRedshiftString)
        # Add to galaxies data array
        if self.galaxies is None:
            self.galaxies = np.copy(OUTPUT.galaxies)
        else:
            self.galaxies = np.append(self.galaxies,np.copy(OUTPUT.galaxies))
        # Delete output class
        del OUTPUT
        # Report progress
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line(task="Redshift = "+str(z))
        return 
    
    def selectOutput(self,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if self.outputs is None:
            return None
        # Select epoch closest to specified redshift        
        iselect = np.argmin(np.fabs(self.outputs.z-z))
        outstr = "Output"+str(self.outputs["iout"][iselect])
        if self._verbose:
            print(funcname+"(): Nearest output is "+outstr+" (redshift = "+str(self.outputs.z[iselect])+")")
        return self.fileObj["Outputs/"+outstr]

    def selectOutputsInRedshiftRange(self,zlow=None,zupp=None,lightconeRedshifts=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        z = np.sort(np.copy(self.outputs.z))
        if zlow is not None:
            ilow = np.argwhere(z>=zlow)[0][0]
            if lightconeRedshifts:
                ilow -= 1
            ilow = np.maximum(ilow,0)
            z = z[ilow:]
        if zupp is not None:
            iupp = np.argwhere(z<=zupp)[-1][0]
            if lightconeRedshifts:
                iupp += 1
            iupp = np.minimum(iupp,len(z))
            z = z[:iupp+1]
        return z
        
    def calculateProperties(self,galaxyProperties,z,overwrite=False):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        

        # Galaxy inclination
        if "inclination" in galaxyProperties:            
            from .Inclination import getInclination
            if self._verbose:
                print(funcname+"(): calculating inclination (z="+str(z)+")")
            getInclination(self,z,overwrite=overwrite,returnDataset=False)
        # Total stellar mass
        if "totalMassStellar" in galaxyProperties:
            from .Stars import getStellarMass
            if self._verbose:
                print(funcname+"(): calculating massStellar (z="+str(z)+")")
            getStellarMass(self,z,"totalMassStellar",overwrite=overwrite,returnDataset=False)
        # Total star formation rate
        if "totalStarFormationRate" in galaxyProperties:
            from .Stars import getStarFormationRate
            if self._verbose:
                print(funcname+"(): calculating starFormationRate (z="+str(z)+")")
            getStarFormationRate(self,z,"totalStarFormationRate",overwrite=overwrite,returnDataset=False)

        return





      
class SnapshotOutput(object):
    
    def __init__(self,redshift,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Store GalacticusHDF5 object
        self.galHDF5Obj = galHDF5Obj
        # Select redshift and output
        self.redshift = self.galHDF5Obj.nearestRedshift(redshift)
        self.redshiftString = self.galHDF5Obj.getRedshiftString(self.redshift)
        self.out = self.galHDF5Obj.selectOutput(self.redshift)
        # Count number of galaxies
        self.numberGalaxies = self.galHDF5Obj.countGalaxiesAtRedshift(self.redshift)
        return
    

    def getGalaxyDataset(self,datasetName,SIunits=False,dataTypeName=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set name to store galaxy dataset under
        if dataTypeName is None:
            dataTypeName = datasetName
        # Store galaxy data
        if datasetName in self.galHDF5Obj.availableDatasets(self.redshift):
            self.galaxies[dataTypeName] = np.copy(np.array(self.out["nodeData/"+datasetName]))
            if SIunits:
                if "unitsInSI" in out["nodeData/"+p].attrs.keys():
                    unit = out["nodeData/"+p].attrs["unitsInSI"]
                    self.galaxies[dataTypeName] *= unit
        # Special cases!
        if datasetName in special_cases:
            if datasetName in ["weight","mergerTreeWeight"]:
                cts = np.array(self.out["mergerTreeCount"])
                wgt = np.array(self.out["mergerTreeWeight"])
                self.galaxies[dataTypeName] = np.copy(np.repeat(wgt,cts))
                del cts,wgt
            if datasetName == "snapshotRedshift":
                self.galaxies[dataTypeName] = np.ones_like(self.galaxies[dataTypeName])*self.redshift
            if datasetName in ["lightconeRightAscension","lightconeDeclination"]:
                available = list(set(["lightconePositionX","lightconePositionY","lightconePositionZ"]).intersection(self.galHDF5Obj.availableDatasets(self.redshift)))
                if len(available) != 3:
                    print("WARNING! "+funcname+"(): at one of lightconePosition[XYZ] not found -- unable to compute "+datasetName+"!")
                    self.galaxies[dataTypeName] = np.ones_like(self.galaxies[dataTypeName])*999.9
                else:
                    rightAscension,declination = getRaDec(np.array(self.out["nodeData/lightconePositionX"]),np.array(self.out["nodeData/lightconePositionY"]),\
                                                              np.array(self.out["nodeData/lightconePositionZ"]),degrees=True)
                    if datasetName == "lightconeRightAscension":
                        self.galaxies[dataTypeName] = np.copy(rightAscension)
                    if datasetName == "lightconeDeclination":
                        self.galaxies[dataTypeName] = np.copy(declination)
                    del rightAscension,declination
        return


    def _getDataTypeNames(self,prop):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        prop = prop.replace(":"+self.redshiftString,"")
        allpropsZ = self.galHDF5Obj.availableDatasets(self.redshift)
        allprops = [p.replace(":"+self.redshiftString,"") for p in allpropsZ]
        matches = fnmatch.filter(allprops,prop) + fnmatch.filter(allprops,prop) + fnmatch.filter(special_cases,prop)
        matches = [m.replace(":"+self.redshiftString,"") for m in matches]
        matches = list(np.unique(matches))
        dummy = [self._dataTypeNames.append(m) for m in matches]
        return
    
    def _findDatasetName(self,prop):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if prop in special_cases:
            self._datasetNames.append(prop)
            return        
        allpropsZ = self.galHDF5Obj.availableDatasets(self.redshift)
        allprops = [p.replace(":"+self.redshiftString,"") for p in allpropsZ]              
        index = np.where(np.array(allprops)==prop)[0][0]
        self._datasetNames.append(allpropsZ[index])
        return
    
    def getDatasetType(self,datasetName,dataTypeName=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if datasetName not in self.galHDF5Obj.availableDatasets(self.redshift):
            raise KeyError(funcname+"(): dataset '"+datasetName+"' not found in file '"+self.galHDF5Obj.fileObj.filename+"'!")
        # Get datatype
        dtype = getDataType(self.out["nodeData/"+datasetName])
        # Select name to use in datatype
        if dataTypeName is None:
            dataTypeName = datasetName
        return (dataTypeName,dtype)
    
    def _buildDataType(self,datasetName,dataTypeName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if datasetName in special_cases:
            self._dtype.append((dataTypeName,float))
            return
        self._dtype.append(self.getDatasetType(datasetName,dataTypeName=dataTypeName))
        return
                                   
    def readGalaxies(self,props=None,SIunits=False,removeRedshiftString=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get list of datasets and corresponding names in file
        self._dataTypeNames = []
        dummy = [self._getDataTypeNames(prop) for prop in props]
        self._datasetNames = []
        dummy = [self._findDatasetName(prop) for prop in self._dataTypeNames]
        if not removeRedshiftString:
            self._dataTypeNames = self._datasetNames
        # Build galaxies array
        self._dtype = []
        dummy = [self._buildDataType(self._datasetNames[i],dataTypeName=self._dataTypeNames[i])\
                     for i in range(len(self._dataTypeNames))]
        self.galaxies = np.zeros(self.numberGalaxies,dtype=self._dtype)
        # Read properties        
        dummy = [self.getGalaxyDataset(self._datasetNames[i],SIunits=False,dataTypeName=self._dataTypeNames[i])\
                     for i in range(len(self._datasetNames))]
        del self._datasetNames,self._dataTypeNames,self._dtype
        return
        

    
    
    


    
