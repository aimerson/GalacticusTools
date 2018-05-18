#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .galaxyProperties import DatasetClass
from .GalacticusErrors import ParseError
from .utils.progress import Progress
from .Inclination import getInclination
from .constants import metallicitySolar,Pi
from .constants import gravitationalConstant,massSolar,megaParsec,speedOfLight
from .constants import plancksConstant,erg,massHydrogen


def parseColdGas(datasetName):
    funcname = sys._getframe().f_code.co_name
    if "abundances" in datasetName.lower():
        searchString = "^(?P<component>disk|spheroid|total)Abundances(?P<gas>Cold)?GasMetals"
    else:
        searchString = "^(?P<component>disk|spheroid|total)(?P<property>Mass|Metallicity)(?P<gas>Cold)?Gas"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH


def parseHIGas(datasetName):
    funcname = sys._getframe().f_code.co_name
    searchString = "^(?P<component>disk|spheroid|total)(?P<property>Mass|Metallicity|Luminosity)HIGas"+\
        "(?P<optionString>:(?P<peak>variable|fixed)Peak)?"
    MATCH = re.search(searchString,datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    return MATCH
    


class GasMassClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,mass=None):
        super(GasMassClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.mass = mass
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.mass = None
        return

class GasMetallicityClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,metallicity=None):
        super(GasMassClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.metallicity = metallicity
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.metallicity = None
        return

class GasLuminosityClass(DatasetClass):
    
    def __init__(self,datasetName=None,redshift=None,outputName=None,luminosity=None):
        super(GasMassClass,self).__init__(datasetName=datasetName,redshift=redshift,outputName=outputName)
        self.luminosity = luminosity
        return

    def reset(self):
        self.datasetName = None
        self.redshift = None
        self.outputName = None
        self.luminosity = None
        return


class ColdGas(object):

    def __init__(self,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.galHDF5Obj = galHDF5Obj
        return
    
    def createColdGasClass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store gas information
        if "mass" in datasetName.lower():
            GAS = GasMassClass()
        elif "abundance" in datasetName.lower():
            GAS = GasMassClass()
        elif "metal" in datasetName.lower():
            GAS = GasMetallicityClass()
        else:
            raise ValueError(funcname+"(): unable to determine type of gas class to create for property '"+datasetName+"'.")
        GAS.datasetName = parseGas(datasetName)
        # Identify HDF5 output
        GAS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            GAS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(redshift)
            GAS.redshift = np.ones(ngals,dtype=float)*redshift
        return GAS
    
    def setColdGasMass(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create gas class
        GAS = self.createColdGasClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(GAS.datasetName.group('redshift'))) and not overwrite:
            GAS.mass = np.array(HDF5OUT["nodeData/"+datasetName])
            return GAS
        # Compute gas mass
        if fnmatch.fnmatch(GAS.datasetName.group('component'),"total"):
            GAS.mass = np.array(HDF5OUT["nodeData/spheroidMassGas"]) + \
                np.array(HDF5OUT["nodeData/diskMassGas"])
        else:
            GAS.mass = np.array(HDF5OUT["nodeData/"+GAS.datasetName.group('component')+"MassGas"])
        return GAS

    def getColdGasMass(self,datasetName,z,overwrite=False):
        GAS = self.setColdGasMass(datasetName,z,overwrite=overwrite)
        return GAS.mass

    def setColdGasMetalsAbundance(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create gas class
        GAS = self.createColdGasClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(GAS.datasetName.group('redshift'))) and not overwrite:
            GAS.mass = np.array(HDF5OUT["nodeData/"+datasetName])
            return GAS
        # Compute gas metals mass
        if fnmatch.fnmatch(GAS.datasetName.group('component'),"total"):
            GAS.mass = np.array(HDF5OUT["nodeData/spheroidAbundancesGasMetals"]) + \
                np.array(HDF5OUT["nodeData/diskAbundancesGasMetals"])
        else:
            GAS.mass = np.array(HDF5OUT["nodeData/"+GAS.datasetName.group('component')+"AbundancesGasMetals"])
        return GAS

    def getColdGasMetalsAbundance(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        GAS = self.setColdGasMetalsAbundance(datasetName,z,overwrite=overwrite)
        return GAS.mass

    
    def setColdGasMetallicity(self,datasetName,z,overwrite=False,solarUnits=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create gas class
        GAS = self.createColdGasClass(datasetName)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(GAS.datasetName.group('redshift'))) and not overwrite:
            GAS.mass = np.array(HDF5OUT["nodeData/"+datasetName])
            return GAS
        # Compute metallicity
        ABUND = self.getColdGasMetalsAbundance(GAS.datasetName.group('component')+"AbundancesGasMetals",z=z)
        COLD = self.getColdGasMass(GAS.datasetName.group('component')+"MassColdGas",z=z)
        GAS.metallicity = np.copy(ABUND.mass/COLD.mass)
        del ABUND,COLD
        if solarUnits:
            GAS.metallicity /= metallicitySolar    
        return GAS

    def getColdGasMetallicity(datasetName,z,overwrite=False,solarUnits=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        GAS = self.setColdGasMetallicity(datasetName,z,overwrite=overwrite,solarUnits=solarUnits)
        return GAS.metallicity
    
    
class HIGas(ColdGas):

    def __init__(self,galHDF5Obj):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        super(HIGas,self).__init__(galHDF5Obj)
        return

    def createHIGasClass(self,datasetName,z):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create class to store dust optical depths information
        if "mass" in datasetName.lower():
            GAS = GasMassClass()
        elif "metal" in datasetName.lower():
            GAS = GasMetallicityClass()
        elif "luminosity" in datasetName.lower():
            GAS = GasLuminosityClass()
        else:
            raise ValueError(funcname+"(): unable to determine type of gas class to create for property '"+datasetName+"'.")
        GAS.datasetName = parseGas(datasetName)
        # Identify HDF5 output
        GAS.outputName = self.galHDF5Obj.nearestOutputName(z)
        # Set redshift
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if "lightconeRedshift" in HDF5OUT.keys():
            GAS.redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
            ngals = self.galHDF5Obj.countGalaxiesAtRedshift(redshift)
            GAS.redshift = np.ones(ngals,dtype=float)*redshift
        return GAS

    def computeRgal(self,component,z,pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        K = gravitationalConstant/(8.0*Pi*pressureInStars)        
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        scaleRadius = np.array(HDF5OUT["nodeData/"+component+"Radius"])*megaParsec/1.68
        COLD = self.getColdGasMass(component+"MassGas",z=z)        
        massStellar = np.array(HDF5OUT["nodeData/"+component+"MassStellar"])
        massFactor = COLD.mass(COLD.mass+meanDispersionRatio*MassStellar)*(massSolar**2)
        Rmol = (K*(scaleRadius**-4)*massFactor)**0.8
        return Rgal

    def computeVariablePeakHIMass(self,component,z,pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        Rgal = self.computeRgal(component,z,pressureInStars=pressureInStars,meanDispersionRatio=meanDispersionRatio)
        COLD = self.getColdGasMass(component+"MassGas",z=z)        
        METALS = self.getColdGasMetalsAbundance(component+"AbundancesGasMetals",z=z)            
        massH = (COLD.mass-METALS.mass)/(1.0+ratioHeH)
        mass = np.copy(massH/(1.0+Rgal))                
        return mass
        
    def computeHIMass(self,GAS,z,hiMassFraction=0.54,Rgal=0.4,ratioHeH=0.73,\
                          pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        gasName = GAS.datasetName.group('component')+"MassGas"
        abundanceName = GAS.datasetName.group('component')+"AbundancesGasMetals"
        # Options for computing HI mass:
        if GAS.datasetName.group('optionString') is None:
            COLD = self.getColdGasMass(gasName,z=z)        
            # Simple assumption that HI mass = cold gas mass * hiMassFraction
            GAS.mass = np.copy(COLD.mass*hiMassFraction)
            del COLD
        else:         
            # More complex options.
            # Paper references:
            # http://adsabs.harvard.edu/abs/2010MNRAS.406...43P (fixed peak case)
            # http://adsabs.harvard.edu/abs/2009ApJ...698.1467O
            # http://adsabs.harvard.edu/abs/2009ApJ...696L.129O            
            if fnmatch.fnmatch(GAS.datasetName.group('option').lower(),"fixed"):
                # Assuming fixed peak
                COLD = self.getColdGasMass(gasName,z=z)
                METALS = self.getColdGasMetalsAbundance(abundanceName,z=z)
                massH = (COLD.mass-METALS.mass)/(1.0+ratioHeH)
                GAS.mass = np.copy(massH/(1.0+Rgal))                
                del massHydrogen,METALS,COLD
            else:
                # Assuming variable peak
                kwargs = {}
                kwargs["pressureInStars"] = pressureInStars
                kwargs["meanDispersionRatio"] = meanDispersionRatio
                if fnmatch.fnmatch(GAS.datasetName.group('component'),"total"):                    
                    mass = self.computeVariablePeakHIMass("disk",z,**kwargs) + \
                        self.computeVariablePeakHIMass("spheroid",z,**kwargs)
                else:
                    mass = self.computeVariablePeakHIMass(GAS.datasetName.group('component'),z,**kwargs)
                GAS.mass = np.copy(mass)
                del mass
        return GAS            
            
    def setHIMass(self,datasetName,z,overwrite=False,hiMassFraction=0.54,Rgal=0.4,ratioHeH=0.73,\
                      pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create gas class
        GAS = self.createGasClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(GAS.datasetName.group('redshift')))
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(GAS.datasetName.group('redshift'))) and not overwrite:
            GAS.mass = np.array(HDF5OUT["nodeData/"+datasetName])
            return GAS
        # Compute gas mass
        kwargs = {}
        kwargs["hiMassFraction"] = hiMassFraction
        kwargs["Rgal"] = Rgal
        kwargs["ratioHeH"] = ratioHeH
        kwargs["pressureInStars"] = pressureInStars
        kwargs["meanDispersionRatio"] = meanDispersionRatio
        GAS = self.computeHIMass(GAS,z,**kwargs)
        return GAS

    def getHIGasMass(self,datasetName,z,overwrite=False,hiMassFraction=0.54,Rgal=0.4,ratioHeH=0.73,\
                         pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        kwargs = {}
        kwargs["hiMassFraction"] = hiMassFraction
        kwargs["Rgal"] = Rgal
        kwargs["ratioHeH"] = ratioHeH
        kwargs["pressureInStars"] = pressureInStars
        kwargs["meanDispersionRatio"] = meanDispersionRatio
        GAS = self.setGasMass(datasetName,z,overwrite=overwrite,**kwargs)
        return GAS.mass
    
    def getLineWidth21cm(self,datasetName,z,overwrite=False):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check dataset name is a line width
        searchString = "^(?P<component>disk|spheroid)LineWidth21cm(?P<averaged>:sphericalAverage)?"
        MATCH = re.search(searchString,datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        # Check if dataset already exists
        if datasetName in self.galHDF5Obj.availableDatasets(z) and not overwrite:
            lineWidth = np.array(HDF5OUT["nodeData/"+datasetName])
            return lineWidth
        # Compute line width
        restWavelength = 21.0e-2 # 21cm in m
        restFrequency21cm = (speedOfLight/restWavelength)/1.0e6 # MHz
        # Compute disk velocity line width
        velocity = np.copy(np.array(HDF5OUT["nodeData/"MATCH.group('component')+"Velocity"]))
        HDF5OUT = self.galHDF5Obj.selectOutput(z)
        if MATCH.group('averaged') is not None:
            velocityLineWidth = 1.57*velocity
        else:
            inclination = getInclination(self.galHDF5Obj,z)
            velocityLineWidth = 2.0*velocity*np.sin(inclination*Pi/180.0)
        velocityLineWidth *= 1000.0 # km --> m
        # Compute line width in frequency space
        lineWidth = velocityLineWidth*restFrequency21cm/speedOfLight
        # Apply (1+z) factor
        if "lightconeRedshift" in HDF5OUT.keys():
            redshift = np.array(HDF5OUT["nodeData/lightconeRedshift"])
        else:
            redshift = self.galHDF5Obj.nearestRedshift(z)
        lineWidth /= (1.0+redshift)
        return lineWidth

    def setHILuminosity(self,datasetName,z,overwrite=False,hiMassFraction=0.54,Rgal=0.4,ratioHeH=0.73,\
                            pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        # Paper reference: http://adsabs.harvard.edu/abs/2010MNRAS.406...43P
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Create gas class
        GAS = self.createGasClass(datasetName,z)
        # Identify HDF5 output
        HDF5OUT = self.galHDF5Obj.selectOutput(float(GAS.datasetName.group('redshift')))
        # Check if mass already available
        if datasetName in self.galHDF5Obj.availableDatasets(float(GAS.datasetName.group('redshift'))) and not overwrite:
            GAS.luminosity = np.array(HDF5OUT["nodeData/"+datasetName])
            return GAS        
        # Store key words to pass to subsequent calls
        kwargs = {}
        kwargs["hiMassFraction"] = hiMassFraction
        kwargs["Rgal"] = Rgal
        kwargs["ratioHeH"] = ratioHeH
        kwargs["pressureInStars"] = pressureInStars
        kwargs["meanDispersionRatio"] = meanDispersionRatio
        # Compute HI luminosity
        if fnmatch.fnmatch(GAS.datasetName.group('component'),"total"):
            DISKGAS = self.setHILuminosity(datasetName.replace("total","disk"),z,overwrite=overwrite,**kwargs)
            SPHEREGAS = self.setHILuminosity(datasetName.replace("total","spheroid"),z,overwrite=overwrite,**kwargs)
            GAS.luminosity = np.copy(DISKGAS.luminosity) + np.copy(SPHEREGAS.luminosity)
            del DISKGAS,SPHEREGAS
        else:            
            # Compute HI mass
            HI = self.getHIMass(datasetName.replace("Luminosity","Mass"),z,overwrite=overwrite,**kwargs)
            # Get velocity line width
            lineWidthName = GAS.datasetName.group('component')+"LineWidth21cm"
            velocityLineWidth = self.getLineWidth21cm(lineWidthName,z,overwrite=overwrite)
            velocityLineWidth *= (1.0+GAS.redshift)
            # Compute luminosity
            einsteinA12 = 2.869e-15 # 1/s
            GAS.luminosity = 0.75*plancksConstant*speedOfLight*einsteinA12
            GAS.luminosity *= np.copy(HI.mass)*massSolar/massHydrogen
            GAS.luminosity /= np.copy(velocityLineWidth)
            GAS.luminosity /= erg
        return GAS


    def getHILuminosity(self,datasetName,z,overwrite=False,hiMassFraction=0.54,Rgal=0.4,ratioHeH=0.73,\
                            pressureInStars=2.35e-13,meanDispersionRatio=0.4):
        # Paper reference: http://adsabs.harvard.edu/abs/2010MNRAS.406...43P
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        kwargs = {}
        kwargs["hiMassFraction"] = hiMassFraction
        kwargs["Rgal"] = Rgal
        kwargs["ratioHeH"] = ratioHeH
        kwargs["pressureInStars"] = pressureInStars
        kwargs["meanDispersionRatio"] = meanDispersionRatio
        GAS = self.setHILuminosity(datasetName,z,overwrite=overwrite,**kwargs)
        return GAS.luminosity
        


def massHI(massColdGas,massMetalsGas,diskMassStellar,diskVelocity,diskRadius,fixedPeak=False,\
               ratioHeH=0.73,pressureInStars=2.35e-13,Rgal=0.4,meanDispersionRatio=0.4):
    # Paper references:
    # http://adsabs.harvard.edu/abs/2010MNRAS.406...43P (fixed peak case)
    # http://adsabs.harvard.edu/abs/2009ApJ...698.1467O
    # http://adsabs.harvard.edu/abs/2009ApJ...696L.129O
    if not fixedPeak:
        K = gravitationalConstant/(8.0*Pi*pressureInStars)        
        scaleRadius = diskRadius*megaParsec/1.68    
        massFactor = massColdGas(massColdGas+meanDispersionRatio*diskMassStellar)*(massSolar**2)
        Rmol = (K*(scaleRadius**-4)*massFactor)**0.8
        Rgal = (3.44*(Rmol**-0.506)+4.82*(Rmol**-1.054))**(-1.0)
    massHydrogen = (massColdGas-massMetalasGas)/(1.0+ratioHeH)
    massHI = massHydrogen/(1.0+Rgal)
    return massHI

def luminosityHI(massHI,diskVelocity,inclination=None):
    # Paper reference: http://adsabs.harvard.edu/abs/2010MNRAS.406...43P
    einsteinA12 = 2.869e-15 # 1/s
    # Compute disk velocity line width
    if inclination is None:        
        velocityLineWidth = 1.57*diskVelocity
    else:
        velocityLineWidth = 2.0*diskVelocity*np.sin(inclination*Pi/180.0)
    velocityLineWidth *= 1000.0 # km --> m
    # Compute rest-frame luminosity
    luminosityHI = 0.75*plancksConstant*speedOfLight*einsteinA12
    luminosityHI *= massHI*massSolar/massHydrogen
    luminosityHI /= velocityLineWidth
    # Convert from J/s to erg/s
    luminosityHI /= erg
    return luminosityHI


def lineWidth21cm(diskVelocity,redshift,inclination=None):
    restWavelength = 21.0e-2 # 21cm in m
    restFrequency21cm = (speedOfLight/restWavelength)/1.0e6 # MHz
    # Compute disk velocity line width
    if inclination is None:        
        velocityLineWidth = 1.57*diskVelocity
    else:
        velocityLineWidth = 2.0*diskVelocity*np.sin(inclination*Pi/180.0)
    velocityLineWidth *= 1000.0 # km --> m
    # Compute line width in frequency space
    lineWidth = velocityLineWidth*restFrequency21cm/speedOfLight
    lineWidth /= (1.0+redshift)
    return lineWidth








def getHIGasMass(galHDF5Obj,z,datasetName,hiMassFraction=0.54,overwrite=False,returnDataset=True):
    """
    getHIGasMass: Calculate and store mass of HI gas in galaxy. This property
                    is named (total|disk|spheroid)MassHIGas.

    USAGE:  mass = getColdGasMass(galHDF5Obj,z,datasetName,[hiMassFraction],[overwrite],[returnDataset])

      Inputs

          galHDF5Obj     : Instance of GalacticusHDF5 file object. 
          z              : Redshift of output to work with.
          datasetName    : Name of dataset to return/process, i.e. (disk|spheroid|total)MassHIGas.
          hiMassFraction : Constant mass fraction for HI. (Default value = 0.54 from 
                           Power, Baugh & Lacey; 2009; http://adsabs.harvard.edu/abs/2009arXiv0908.1396P)
          overwrite      : Overwrite any existing value for gas mass. (Default value = False)
          returnDataset  : Return array of dataset values? (Default value = True)

      Outputs

          mass           : Numpy array of gas masses (if returnDataset=True).
                                                                                                                                                                                            
    """
    funcname = sys._getframe().f_code.co_name
    # Check dataset name is a stellar mass
    MATCH = re.search("(\w+)?MassHIGas",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Get galaxy component
    component = MATCH.group(1)
    if component is None:
        component = "total"
    component = component.lower()    
    # Check if gas mass already calculated
    datasetName = component.lower()+"MassHIGas"
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+datasetName])
    # Extract cold gas mass
    if component == "total":
        gasMass = getColdGasMass(galHDF5Obj,z,"totalMassColdGas",overwrite=overwrite,returnDataset=True)
    else:
        props = [component+"MassGas"]
        data = galHDF5Obj.readGalaxies(z,props=props)
        gasMass = np.copy(data[component+"MassGas"])
        del data
    # Compute HI mass
    gasMass *= hiMassFraction
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,gasMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskMassGas"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,{"hiMassFraction":hiMassFraction})
    if returnDataset:
        return gasMass
    return











