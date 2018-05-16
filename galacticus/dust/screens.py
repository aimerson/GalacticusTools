#! /usr/bin/env python

# Values/fits from: http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html

import sys,re,os
import numpy as np
import copy
import fnmatch
import pkg_resources
from scipy.interpolate import interp1d
from .utils import DustProperties,DustClass
from ..GalacticusErrors import ParseError
from ..constants import angstrom,micron,luminosityAB,luminositySolar

########################################################################
# CLASSES FOR DUST SCREEN MODELS
########################################################################

class SCREEN(object):

    def __init__(self,Rv=None,dustTable=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.Rv = Rv
        self.dustTable = dustTable
        self.dustCurve = None
        return

    def updateRv(self,Rv):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.Rv = Rv
        return
    
    def updateDustTable(self,dustTable):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.dustTable = dustTable
        self.setDustCurve()
        return
    
    def setDustCurve(self):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if self.dustTable is None:
            raise ValueError(funcname+"(): dust table has not been specified.")
        self.dustCurve = interp1d(self.dustTable.wavelength,self.dustTable.klambda,kind='linear',fill_value="extrapolate")
        return

    def attenuation(self,wavelength,Av):        
        if self.dustCurve is None:
            self.setDustCurve()
        return self.dustCurve(wavelength*angstrom/micron)*Av


def parseDustAttenuatedLuminosity(datasetName):
    if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
        searchString = "^(?P<component>disk|spheroid|total)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
            "(?P<filterName>:[^:]+)?"
    else:
        searchString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"
    searchString = searchString + "(?P<redshiftString>:z(?P<redshift>[\d\.]+))"
    searchString = searchString + "(?P<dust>:dustScreen_(?P<dustName>[^_]+)(?P<ageString>_age(?P<age>[\d\.]+))?_Av(?P<av>[\d\.]+))$"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


class dustScreen(DustProperties):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set verbosity
        self.verbose = verbose
        # Initialise DustProperties class
        super(dustScreen,self).__init__()
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        return

    def selectDustScreen(self,dustName,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        name = dustName.lower()
        if fnmatch.fnmatch(name,"calzetti*"):
            SCREENOBJ = Calzetti(Rv=Rv)
        elif fnmatch.fnmatch(name,"allen*"):
            SCREENOBJ = Allen(Rv=Rv)
        elif fnmatch.fnmatch(name,"fitzpatrick*"):
            SCREENOBJ = Fitzpatrick(Rv=Rv,galaxy="LMC")
        elif fnmatch.fnmatch(name,"seaton*"):
            SCREENOBJ = Fitzpatrick(Rv=Rv,galaxy="MW")
        elif fnmatch.fnmatch(name,"prevot*"):
            SCREENOBJ = Prevot(Rv=Rv)
        else:
            raise ValueError(funcname+"(): dust screen name '"+dustName+"' not recognized.")
        return SCREENOBJ

    def getEffectiveWavelength(self,DUST):
        redshift = None
        if DUST.datasetName.group('frame').replace(":","") == "observed":
            redshift = DUST.redshift
        if 'lineName' in DUST.datasetName.groupdict().keys():
            name = DUST.datasetName.group('lineName')
            redshift = None
        else:
            name = DUST.datasetName.group('filterName')
        effectiveWavelength = self.effectiveWavelength(name,redshift=redshift,verbose=self.verbose)
        return effectiveWavelength

    def setOpticalDepths(self,DUST,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Select dust screen         
        SCREENOBJ = self.selectDustScreen(DUST.datasetName.group('dustName'),Rv=Rv)
        age = None
        if DUST.datasetName.group('ageString') is not None:
            age = float(DUST.datasetName.group('age'))
        Av = float(DUST.datasetName.group('av'))
        # Get effective wavelength
        effectiveWavelength = self.getEffectiveWavelength(DUST)
        # Apply attenuation
        attenuation = SCREENOBJ.attenuation(effectiveWavelength,Av)
        # Get optical depth
        e = np.exp(1.0)
        if "lineName" in DUST.datasetName.groupdict().keys():
            DUST.opticalDepthClouds = SCREENOBJ.attenuation(effectiveWavelength,Av)/(2.5*np.log10(e))
            DUST.opticalDepthISM = DUST.opticalDepthClouds*0.0
        else:
            DUST.opticalDepthISM = SCREENOBJ.attenuation(effectiveWavelength,Av)/(2.5*np.log10(e))
            DUST.opticalDepthClouds = DUST.opticalDepthISM*0.0        
        return DUST
        
    def createDustClass(self,datasetName):
        # Create class to store dust optical depths information
        DUST = DustClass()
        DUST.datasetName = parseDustAttenuatedLuminosity(datasetName)
        # Identify HDF5 output
        DUST.outputName = self.galHDF5Obj.nearestOutputName(float(DUST.datasetName.group('redshift')))
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        if "lightconeRedshift" in HDF5OUT.keys():
            DUST.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            z = self.galHDF5Obj.nearestRedshift(float(DUST.datasetName.group('redshift')))
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(z)
            DUST.redshift = np.ones(ngals,dtype=float)*z
        return DUST

    def getDustFreeLuminosities(self,DUST):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        # Extract dust free luminosities
        luminosityName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"")
        if "LineLuminosity" in DUST.datasetName.group(0):
            recentName = luminosityName
        else:
            #recentName = DUST.datasetName.group(0).replace(DUST.datasetName.group('dust'),"recent")
            #if not self.galHDF5Obj.datasetExists(recentName,float(DUST.datasetName.group('redshift'))):
            #    raise KeyError(funcname+"(): 'recent' dataset "+recentName+" not found!")
            recentName = None
        luminosity = np.copy(np.array(HDF5OUT["nodeData/"+luminosityName]))
        if recentName is None:
            recentLuminosity = np.zeros_like(luminosity)
        else:
            recentLuminosity = np.copy(np.array(HDF5OUT["nodeData/"+recentName]))
        return (luminosity,recentLuminosity)

    def setAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create dust class
        DUST = self.createDustClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(DUST.datasetName.group('redshift')))
        # Check if luminosity already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(DUST.datasetName.group('redshift'))) and not overwrite:
            DUST.attenuatedLuminosity = np.array(HDF5OUT["nodeData/"+datasetName])
            return DUST
        # Compute optical depth
        DUST = self.setOpticalDepths(DUST,Rv)
        DUST.Rv = Rv
        # Get dust free luminosities
        luminosity,recentLuminosity = self.getDustFreeLuminosities(DUST)
        # Store luminosity
        DUST.attenuatedLuminosity = DUST.attenuate(luminosity,recentLuminosity)
        return DUST

    def getAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        DUST = self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,Rv=Rv)
        return DUST

    def writeLuminosityToFile(self,datasetName,z=None,overwrite=False,Rv=None):
        DUST = self.getAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,Rv=Rv)
        redshift = float(DUST.datasetName.group('redshift'))
        if not DUST.datasetName.group(0) in self.galHDF5Obj.availableDatasets(redshift) or overwrite:
            # Select HDF5 output
            HDF5OUT = self.galHDF5Obj.selectOutput(redshift)
            # Add luminosity to file
            self.galHDF5Obj.addDataset(HDF5OUT.name+"/nodeData/",DUST.datasetName.group(0),np.copy(DUST.attenuatedLuminosity))
            # Add appropriate attributes to new dataset
            if fnmatch.fnmatch(self.datasetName.group(0),"*LineLuminosity*"):
                attr = {"unitsInSI":luminositySolar}
            else:
                attr = {"unitsInSI":luminosityAB}
            self.galHDF5Obj.addAttributes(out.name+"/nodeData/"+self.datasetName.group(0),attr)
        return




class Allen(SCREEN):
    
    def __init__(self,Rv=3.1):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if Rv is None:
            Rv = 3.1
        # Construct array for dust curve
        dustTable = self.createDustTable(Rv)
        # Initalise SCREEN class
        super(Allen,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Allen (1976) [MW]"
        return

    def createDustTable(self,Rv):
        # Construct array for dust curve
        dustFile = pkg_resources.resource_filename(__name__,"data/dust/screens/dustTableAllen1976.dat")
        dustFile = dustFile.replace("/dust/data","/data")        
        if not os.path.exists(dustFile):
            self.writeDustTable(dustFile)
        dustTable = np.loadtxt(dustFile,dtype=[("wavelength",float),("klambda",float)],usecols=[0,1]).view(np.recarray)
        dustTable.wavelength *= angstrom/micron
        return dustTable

    def writeDustTable(self,ofile):
        wavelengths = np.array([1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500., \
                                    2850., 3330., 3650., 4000., 4400., 5000., 5530., 6700., \
                                    9000., 10000., 20000., 100000.])
        klambda = np.array([4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 1.97, 1.69,\
                                1.58, 1.45, 1.32, 1.13, 1.00, 0.74, 0.46, 0.38, 0.11,0.00])    
        f = open(ofile,'w')
        f.write("# wavelength klambda\n")
        fmt = "%7.1f  %6.3f"
        np.savetxt(f,np.c_[wavelengths,klambda],fmt=fmt)
        f.close()
        return        

    
class Calzetti(SCREEN):
    
    def __init__(self,Rv=4.05):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Get dust table
        if Rv is None:
            Rv = 4.05
        dustTable = self.createDustTable(Rv)
        # Initalise SCREEN class
        super(Calzetti,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Calzetti et al. (2000) [SB]"
        return

    def createDustTable(self,Rv):
        diff = 1.0*angstrom/micron
        wavelengths = np.arange(0.12,2.20+diff,diff)
        dustTable = np.zeros(len(wavelengths),dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)
        lower = 2.659*(-2.156+(1.509/wavelengths)-(0.198/wavelengths**2)+(0.011/wavelengths**3))
        upper = 2.659*(-1.857 + (1.040/wavelengths))
        mask = dustTable.wavelength >= 0.63
        dustTable.klambda = np.copy(lower)
        np.place(dustTable.klambda,mask,np.copy(upper[mask]))
        dustTable.klambda += Rv
        dustTable.klambda /= Rv
        del wavelengths    
        return dustTable



class Fitzpatrick(SCREEN):

    def __init__(self,galaxy="MW",Rv=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        self.assertGalaxyType(galaxy)
        # Check value for Rv
        if Rv is None:
            Rv = self.getDefaultRv(galaxy)
        # Set reference
        self.reference = "Fitzpatrick & Massa (1986) ["+galaxy.upper()+"]"      
        # Get dust table
        dustTable = self.createDustTable(Rv,galaxy="MW")
        # Initalise SCREEN class
        super(Fitzpatrick,self).__init__(Rv=Rv,dustTable=dustTable)        
        return

    def getDefaultRv(self,galaxy):
        self.assertGalaxyType(galaxy)
        RV = {"MW":3.1,"LMC":2.72}
        return RV[galaxy.upper()]

    def assertGalaxyType(self,galaxy):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if galaxy.upper() not in ["MW","LMC"]:
            raise KeyError(funcname+"(): galaxy type not recognized. Should be 'MW' or 'LMC'.")
        return

    def colorRatio(self,wavelength,galaxy="MW"):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Colour ratio parameters
        galaxies = {}
        galaxies["MW"] = {"invLambda0":4.595,"gamma":1.051,"C1":-0.38,"C2":0.74,"C3":3.96,"C4":0.26}
        galaxies["LMC"] = {"invLambda0":4.608,"gamma":0.994,"C1":-0.69,"C2":0.89,"C3":2.55,"C4":0.50}
        # Store array of inverted wavelength
        invLambda = 1.0/wavelength
        # Extract arrays of colour ratio parameters
        self.assertGalaxyType(galaxy)
        params = galaxies[galaxy]
        invLambda0 = params["invLambda0"]*np.ones_like(wavelength)
        gamma = params["gamma"]*np.ones_like(wavelength)
        C1 = params["C1"]*np.ones_like(wavelength)
        C2 = params["C2"]*np.ones_like(wavelength)
        C3 = params["C3"]*np.ones_like(wavelength)
        C4 = params["C4"]*np.ones_like(wavelength)
        mask = invLambda < 5.9
        np.place(C4,mask,0.0)
        # Compute colour ratio
        factor2 = C2*invLambda
        factor3 = C3/((invLambda-(invLambda0**2/invLambda))**2+gamma**2)
        factor4 = C4*(0.539*(invLambda-5.9)**2+0.0564*(invLambda-5.9)**3)
        ratio = C1+factor2+factor3+factor4
        return ratio
    
    def createDustTable(self,Rv=None,galaxy="MW"):
        self.assertGalaxyType(galaxy)
        # Check value for Rv
        if Rv is None:
            Rv = self.getDefaultRv(galaxy)
        # Set reference
        self.reference = "Fitzpatrick & Massa (1986) ["+galaxy.upper()+"]"      
        # Get wavelength range
        upperLimit = {"MW":3650,"LMC":3330.0}
        diff = 1.0*angstrom/micron
        low = 1200.0
        upp = upperLimit[galaxy.upper()]
        wavelengths = np.arange(low*angstrom/micron,(upp*angstrom/micron)+diff,diff)
        # Compute dust table using colour ratio
        klambda = self.colorRatio(wavelengths,galaxy) + Rv
        # Build dust table (using Allen et al. dust curve for long wavelengths)
        ALLEN = Allen(Rv)        
        mask = ALLEN.dustTable.wavelength > wavelengths.max()                   
        N = len(wavelengths)+len(ALLEN.dustTable.wavelength[mask])
        dustTable = np.zeros(N,dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.append(np.copy(wavelengths),np.copy(ALLEN.dustTable.wavelength[mask])*angstrom/micron)
        dustTable.klambda = np.append(np.copy(klambda),np.copy(ALLEN.dustTable.klambda[mask]*Rv))
        dustTable.klambda /= Rv
        del wavelengths,klambda
        return dustTable


    
class Prevot(SCREEN):
    
    def __init__(self,Rv=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if Rv is None:
            Rv = 3.1
        # Get dust table
        dustTable = self.createDustTable(Rv)
        # Initalise SCREEN class
        super(Prevot,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Prevot et al. (1984) [SMC]"             
        return

    def createDustTable(self,Rv):
        # Construct array for dust curve
        dustFile = pkg_resources.resource_filename(__name__,"data/dust/screens/dustTablePrevot1984.dat")
        dustFile = dustFile.replace("/dust/data","/data")        
        if not os.path.exists(dustFile):
            self.writeDustTable(dustFile)
        dustTable = np.loadtxt(dustFile,dtype=[("wavelength",float),("klambda",float)],usecols=[0,1]).view(np.recarray)
        dustTable.wavelength *= angstrom/micron
        dustTable.klambda += Rv
        dustTable.klambda /= Rv
        return dustTable

    def writeDustTable(self,ofile):
        wavelengths = np.array([1275., 1330., 1385., 1435., 1490., 1545., 1595., 1647., 1700.,\
                                   1755., 1810., 1860., 1910., 2000., 2115., 2220., 2335., 2445.,\
                                   2550., 2665., 2778., 2890., 2995., 3105., 3704., 4255., 5291.,\
                                   12500., 16500., 22000.])
        klambda = np.array([13.54, 12.52, 11.51, 10.80, 9.84, 9.28, 9.06, 8.49, 8.01, 7.71, 7.17, \
                                6.90, 6.76, 6.38, 5.85, 5.30, 4.53, 4.24, 3.91, 3.49, 3.15, 3.00, \
                                2.65, 2.29, 1.81, 1.00, 0.00, -2.02, -2.36, -2.47])
        f = open(ofile,'w')
        f.write("# wavelength klambda\n")
        fmt = "%7.1f  %6.3f"
        np.savetxt(f,np.c_[wavelengths,klambda],fmt=fmt)
        f.close()
        return        




def screenModel(model,Rv=None):
    
    if fnmatch.fnmatch(model.lower(),"cal*"):
        DUST = Calzetti(Rv=Rv)
    elif fnmatch.fnmatch(model.lower(),"al*"):
        DUST = Allen(Rv=Rv)
    elif fnmatch.fnmatch(model.lower(),"fit*"):
        DUST = Fitzpatrick(Rv=Rv,galaxy="LMC")
    elif fnmatch.fnmatch(model.lower(),"seat*"):
        DUST = Fitzpatrick(Rv=Rv,galaxy="MW")
    elif fnmatch.fnmatch(model.lower(),"prev*"):
        DUST = Prevot(Rv=Rv)
    return DUST
    




