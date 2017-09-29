#! /usr/bin/env python

import sys,fnmatch,re
import numpy as np
from .io import GalacticusHDF5
from .GalacticusErrors import ParseError
from .utils.progress import Progress
from .constants import metallicitySolar,Pi
from .constants import gravitationalConstant,massSolar,megaParsec,speedOfLight
from .constants import plancksConstant,erg,massHydrogen




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





def getColdGasMass(galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
    """
    getColdGasMass: Calculate and store mass of cold gas in galaxy. This property
                    is named (total|disk|spheroid)Mass(Cold)Gas.

    USAGE:  mass = getColdGasMass(galHDF5Obj,z,datasetName,[overwrite],[returnDataset],[progressObj])

      Inputs

          galHDF5Obj     : Instance of GalacticusHDF5 file object. 
          z              : Redshift of output to work with.
          datasetName    : Name of dataset to return/process, i.e. (disk|spheroid|total)Mass(Cold)Gas.
          overwrite      : Overwrite any existing value for gas mass. (Default value = False)
          returnDataset  : Return array of dataset values? (Default value = True)
          progressObj    : Progress object instance to display progress bar if call is inside loop.                                                                 
                           If None, then progress not displayed. (Default value = None) 

      Outputs

          mass           : Numpy array of gas masses (if returnDataset=True).

    """
    funcname = sys._getframe().f_code.co_name
    # Check dataset name is a stellar mass
    MATCH = re.search("(\w+)?Mass(\w+)?Gas",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Get galaxy component
    component = MATCH.group(1)
    if component is None:
        component = "total"
    component = component.lower()    
    # Get gas to calculate
    gas = MATCH.group(2)
    if gas is None:
        gas = ""    
    # Check if simply return disk/spheroid (cold) gas mass
    if component != "total":
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+component+"MassGas"])
    # Check if gas mass already calculated
    datasetName = component.lower()+"Mass"+gas+"Gas"
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+datasetName])
    # Extract stellar mass components and calculate total mass
    data = galHDF5Obj.readGalaxies(z,props=["spheroidMassGas","diskMassGas"])
    totalGasMass = data["diskMassGas"] + data["spheroidMassGas"]
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,totalGasMass,overwrite=overwrite)
    # Extract appropriate attributes and write to new dataset
    attr = out["nodeData/diskMassGas"].attrs
    galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return totalGasMass
    return



def computeColdGasMetallicity(galHDF5Obj,z,datasetName,solarUnits=True):
    funcname = sys._getframe().f_code.co_name
    # Check dataset name is a metallicity
    MATCH = re.search("(\w+)?Metallicity(\w+)?Gas",datasetName)
    if not MATCH:
        raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
    # Get nearest redshift output
    out = galHDF5Obj.selectOutput(z)
    # Get galaxy component
    component = MATCH.group(1)
    if component is None:
        component = "total"
    component = component.lower()
    # Get gas to calculate
    gas = MATCH.group(2)
    if gas is None:
        gas = ""
    # Compute metallicity
    if component != "total":
        abundance = np.copy(np.array(out["nodeData/"+component+"AbundancesGasMetals"]))
        mass = np.copy(np.array(out["nodeData/"+component+"MassGas"]))
    else:
        abundance = np.copy(np.array(out["nodeData/diskAbundancesGasMetals"])) + \
            np.copy(np.array(out["nodeData/spheroidAbundancesGasMetals"]))
        mass = np.copy(np.array(out["nodeData/diskMassGas"])) + \
            np.copy(np.array(out["nodeData/spheroidMassGas"]))            
    result = np.copy(abundance/mass)
    del abundance,mass
    if solarUnits:
        result /= metallicitySolar    
    return result


def getColdGasMetallicity(galHDF5Obj,z,datasetName,solarUnits=True,overwrite=False,returnDataset=True,progressObj=None):
    """
    getColdGasMass: Calculate and store metallicty of cold gas in galaxy. This property
                    is named (total|disk|spheroid)Metallicity(Cold)Gas.

    USAGE:  mass = getColdGasMetallicty(galHDF5Obj,z,datasetName,[overwrite],[returnDataset],[progressObj])

      Inputs

          galHDF5Obj     : Instance of GalacticusHDF5 file object. 
          z              : Redshift of output to work with.
          datasetName    : Name of dataset to return/process, i.e. (disk|spheroid|total)Mass(Cold)Gas.
          solarUnits     : Return metallicty in units of Solar metallicty? (Default = True)
          overwrite      : Overwrite any existing value for gas mass. (Default value = False)
          returnDataset  : Return array of dataset values? (Default value = True)
          progressObj    : Progress object instance to display progress bar if call is inside loop.                                                                 
                           If None, then progress not displayed. (Default value = None) 

      Outputs

          metallicty     : Numpy array of gas metallicity (if returnDataset=True).

    """
    funcname = sys._getframe().f_code.co_name
    # Check if gas metallicty already calculated
    if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if not returnDataset:
            return
        else:
            return np.array(out["nodeData/"+datasetName])
    # Compute metallicity 
    metallicity = computeColdGasMetallicity(galHDF5Obj,z,datasetName,solarUnits=solarUnits)
    # Write dataset to file
    galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,metallicty,overwrite=overwrite)
    if progressObj is not None:
        progressObj.increment()
        progressObj.print_status_line()
    if returnDataset:
        return metallicity
    return



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











