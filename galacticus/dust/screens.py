#! /usr/bin/env python

# Values/fits from: http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html

import sys,re
import numpy as np
import copy
import fnmatch
from scipy.interpolate import interp1d
from .utils import DustProperties
from ..GalacticusErrors import ParseError
from ..constants import angstrom,micron,luminosityAB,luminositySolar

########################################################################
# CLASSES FOR DUST SCREEN MODELS
########################################################################

class SCREEN(object):

    def __init__(self,Rv=None,dustTable=None):
        self.Rv = Rv
        self._dustTable = dustTable
        self.dustCurve = interp1d(self._dustTable.wavelength,self._dustTable.klambda,kind='linear',fill_value="extrapolate")                
        return

    def updateRv(self,Rv):
        self.Rv = Rv
        return

    def attenuation(self,wavelength,Av):        
        return self.dustCurve(wavelength*angstrom/micron)*Av



class dustScreen(DustProperties):
    
    def __init__(self,galHDF5Obj,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set verbosity
        self.verbose = verbose
        # Initialise DustProperties class
        super(dustScreen,self).__init__()
        # Store screen classes
        self.screenObjs = {}        
        # Initialise variables to store Galacticus HDF5 objects
        self.galHDF5Obj = galHDF5Obj
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        # Initialise datset name
        self.datasetName = None
        self.screenOptions = None
        # Set variables to store attenuated luminosity
        self.attenuatedLuminosiy = None
        self.Rv = None
        return

    def resetHDF5Output(self):
        self.redshift = None
        self.hdf5Output = None
        self.redshiftString = None
        return

    def reset(self):
        self.resetAttenuationInformation()        
        return

    def resetAttenuationInformation(self):
        self.attenuatedLuminosity = None
        self.Rv = None
        return

    def setHDF5Output(self,z):
        self.resetHDF5Output()
        self.redshift = self.galHDF5Obj.nearestRedshift(z)
        self.hdf5Output = self.galHDF5Obj.selectOutput(self.redshift)
        return

    def setDatasetName(self,datasetName):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        if fnmatch.fnmatch(datasetName,"*LineLuminosity:*"):
            searchString = "^(?P<component>disk|spheroid|total)LineLuminosity:(?P<lineName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<filterName>:[^:]+)?(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustScreen_(?P<options>[^:]+))$"
        else:
            searchString = "^(?P<component>disk|spheroid|total)LuminositiesStellar:(?P<filterName>[^:]+)(?P<frame>:[^:]+)"+\
                "(?P<redshiftString>:z(?P<redshift>[\d\.]+))(?P<dust>:dustScreen_(?P<options>[^:]+)?)$"
        self.datasetName = re.search(searchString,datasetName)
        if not self.datasetName:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
        optionString = "^(?P<name>[^_]+)(?P<ageString>_age(?P<age>[\d\.]+))?_Av(?P<av>[\d\.]+)"
        self.screenOptions = re.search(optionString,self.datasetName.group('options'))
        return

    def selectDustScreen(self,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        name = self.screenOptions.group('name').lower()
        if name not in self.screenObjs.keys():
            if fnmatch.fnmatch(name,"calzetti*"):                
                self.screenObjs[name] = Calzetti(Rv=self.Rv)
            elif fnmatch.fnmatch(name,"allen*"):                
                self.screenObjs[name] = Allen(Rv=self.Rv)
            elif fnmatch.fnmatch(name,"fitzpatrick*"):                
                self.screenObjs[name] = Fitzpatrick(Rv=self.Rv,galaxy="LMC")
            elif fnmatch.fnmatch(name,"seaton*"):                
                self.screenObjs[name] = Fitzpatrick(Rv=self.Rv,galaxy="MW")
            elif fnmatch.fnmatch(name,"prevot*"):                
                self.screenObjs[name] = Prevot(Rv=self.Rv)
            else:
                raise KeyError(funcname+"(): screen name '"+name+"' not recongised!")
        screenObj = self.screenObjs[name]
        screenObj.updateRv(Rv)
        return screenObj

    def applyAttenuation(self,luminosity,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        screenObj = self.selectDustScreen(Rv=Rv)
        # Get dust screen
        screenObj = self.selectDustScreen(Rv=Rv)
        age = None
        if self.screenOptions.group('ageString') is not None:
            age = float(self.screenOptions.group('age'))
        Av = float(self.screenOptions.group('av'))
        # Get effective wavelength
        effectiveWavelength = self.getEffectiveWavelength()
        # Apply attenuation
        attenuation = screenObj.attenuation(effectiveWavelength,Av)
        # Get optical depth
        e = np.exp(1.0)
        opticalDepth = screenObj.attenuation(effectiveWavelength,Av)/(2.5*np.log10(e))
        attenuation = np.exp(-opticalDepth)        
        return luminosity*attenuation

    def getEffectiveWavelength(self):
        redshift = None
        if self.datasetName.group('frame').replace(":","") == "observed":
            redshift = float(self.redshift)
        if 'lineName' in self.datasetName.re.groupindex.keys():
            name = self.datasetName.group('lineName')
            redshift = None
        else:
            name = self.datasetName.group('filterName')
        effectiveWavelength = self.effectiveWavelength(name,redshift=redshift,verbose=self.verbose)
        return effectiveWavelength

    def setAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check HDF5 snapshot specified
        if z is not None:
            self.setHDF5Output(z)
        else:
            if self.hdf5Output is None:
                z = self.datasetName.group('redshift')
                if z is None:
                    errMsg = funcname+"(): no HDF5 output specified. Either specify the redshift "+\
                        "of the output or include the redshift in the dataset name."
                    raise RunTimeError(errMsg)
                self.setHDF5Output(z)
        # Set datasetName
        self.setDatasetName(datasetName)
        # Reset attenuation information
        self.resetAttenuationInformation()
        self.Rv = Rv
       # Compute attenuated luminosity
        if self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) and not overwrite:
            self.attenuatedLuminosity = np.array(out["nodeData/"+datasetName])
            return
        # Get dust free luminosity
        luminosityName = datasetName.replace(self.datasetName.group('dust'),"")
        luminosity = np.copy(np.array(self.hdf5Output["nodeData/"+luminosityName]))
        # Set luminosity
        self.attenuatedLuminosity = self.applyAttenuation(luminosity)
        return

    def getAttenuatedLuminosity(self,datasetName,overwrite=False,z=None,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.setAttenuatedLuminosity(datasetName,overwrite=overwrite,z=z,Rv=Rv)
        return self.attenuatedLuminosity
   
   
    def writeLuminosityToFile(self,overwrite=False):
        if not self.datasetName.group(0) in self.galHDF5Obj.availableDatasets(self.redshift) or overwrite:
            out = self.galHDF5Obj.selectOutput(self.redshift)
           # Add luminosity to file
            self.galHDF5Obj.addDataset(out.name+"/nodeData/",self.datasetName.group(0),np.copy(self.attenuatedLuminosity))
            # Add appropriate attributes to new dataset               
            if fnmatch.fnmatch(self.datasetName.group(0),"*LineLuminosity*"):
                attr = {"unitsInSI":luminositySolar}
            else:
                attr = {"unitsInSI":luminosityAB}
            if self.Rv is not None:
                attr["Rv"] = self.Rv
            self.galHDF5Obj.addAttributes(out.name+"/nodeData/"+self.datasetName.group(0),attr)
        return



class Allen(SCREEN):
    
    def __init__(self,Rv=3.1):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if Rv is None:
            Rv = 3.1
        # Construct array for dust curve
        wavelengths = np.array([1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500., \
                                    2850., 3330., 3650., 4000., 4400., 5000., 5530., 6700., \
                                    9000., 10000., 20000., 100000.])
        klambda = np.array([4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 1.97, 1.69,\
                                1.58, 1.45, 1.32, 1.13, 1.00, 0.74, 0.46, 0.38, 0.11,0.00])
        dustTable = np.zeros(len(wavelengths),dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)*angstrom/micron
        del wavelengths        
        dustTable.klambda = np.copy(klambda)
        del klambda
        # Initalise SCREEN class
        super(Allen,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Allen (1976) [MW]"
        return

    
class Calzetti(SCREEN):
    
    def __init__(self,Rv=4.05):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if Rv is None:
            Rv = 4.05
        # Construct array for dust curve
        def lowRange(wavelength):
            # Fit for lower regime
            result = 2.659*(-2.156+(1.509/wavelength)-(0.198/wavelength**2)+(0.011/wavelength**3))
            return result 
        def uppRange(wavelength):
            # Fit for upper regime
            result = 2.659*(-1.857 + (1.040/wavelength))
            return result         
        diff = 1.0*angstrom/micron
        wavelengths = np.arange(0.12,2.20+diff,diff)
        dustTable = np.zeros(len(wavelengths),dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)
        del wavelengths
        mask = dustTable.wavelength >= 0.63
        dustTable.klambda = lowRange(dustTable.wavelength) + Rv
        np.place(dustTable.klambda,mask,uppRange(dustTable.wavelength)[mask]+Rv)                
        dustTable.klambda /= Rv
        # Initalise SCREEN class
        super(Calzetti,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Calzetti et al. (2000) [SB]"
        return



class Fitzpatrick(SCREEN):
    def __init__(self,galaxy="MW",Rv=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Construct array for dust curve            
        def colorRatio(wavelength,galaxy):
            invLambda = 1.0/wavelength
            if fnmatch.fnmatch(galaxy.lower(),"mw"):
                # Milky Way
                invLambda0 = 4.595
                gamma = 1.051
                C1 = -0.38
                C2 = 0.74
                C3 = 3.96
                C4 = 0.26
            else:
                # LMC
                invLambda0 = 4.608
                gamma = 0.994
                C1 = -0.69
                C2 = 0.89
                C3 = 2.55
                C4 = 0.50
            if invLambda < 5.9:
                C4 = 0.0
            factor2 = C2*invLambda
            factor3 = C3/((invLambda-(invLambda0**2/invLambda))**2+gamma**2)
            factor4 = C4*(0.539*(invLambda-5.9)**2+0.0564*(invLambda-5.9)**3)
            return C1+factor2+factor3+factor4
        diff = 1.0*angstrom/micron
        low = 1200.0
        if fnmatch.fnmatch(galaxy.lower(),"mw"):      
            self.reference = "Fitzpatrick & Massa (1986) [MW]"      
            upp = 3650
            if Rv is None:
                Rv = 3.1
        elif fnmatch.fnmatch(galaxy.lower(),"lmc"):     
            self.reference = "Fitzpatrick & Massa (1986) [LMC]"             
            upp = 3330
            if Rv is None:
                Rv = 2.72
        wavelengths = np.arange(low*angstrom/micron,(upp*angstrom/micron)+diff,diff)        
        klambda = np.array([colorRatio(l,galaxy) for l in wavelengths]) + Rv
        wavelengthsAllen = np.array([1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500., \
                                         2850., 3330., 3650., 4000., 4400., 5000., 5530., 6700., \
                                         9000., 10000., 20000., 100000.])*angstrom/micron
        klambdaAllen = np.array([4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 1.97, 1.69,\
                                     1.58, 1.45, 1.32, 1.13, 1.00, 0.74, 0.46, 0.38, 0.11,0.00])*Rv
        mask = wavelengthsAllen > wavelengths.max()
        wavelengthsAllen = wavelengthsAllen[mask]
        klambdaAllen = klambdaAllen[mask]
        dustTable = np.zeros(len(wavelengths)+len(wavelengthsAllen),dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)           
        dustTable.wavelength = np.append(np.copy(wavelengths),np.copy(wavelengthsAllen))
        dustTable.klambda = np.append(np.copy(klambda),np.copy(klambdaAllen))
        dustTable.klambda /= Rv
        del wavelengths,klambda
        # Initalise SCREEN class
        super(Fitzpatrick,self).__init__(Rv=Rv,dustTable=dustTable)        
        return
    
    

class Prevot(SCREEN):
    
    def __init__(self,Rv=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        if Rv is None:
            Rv = 3.1
        # Construct array for dust curve
        wavelengths = np.array([1275., 1330., 1385., 1435., 1490., 1545., 1595., 1647., 1700.,\
                                   1755., 1810., 1860., 1910., 2000., 2115., 2220., 2335., 2445.,\
                                   2550., 2665., 2778., 2890., 2995., 3105., 3704., 4255., 5291.,\
                                   12500., 16500., 22000.])
        klambda = np.array([13.54, 12.52, 11.51, 10.80, 9.84, 9.28, 9.06, 8.49, 8.01, 7.71, 7.17, \
                                6.90, 6.76, 6.38, 5.85, 5.30, 4.53, 4.24, 3.91, 3.49, 3.15, 3.00, \
                                2.65, 2.29, 1.81, 1.00, 0.00, -2.02, -2.36, -2.47])
        dustTable = np.zeros(len(wavelengths),dtype=[("wavelength",float),("klambda",float)]).view(np.recarray)
        dustTable.wavelength = np.copy(wavelengths)*angstrom/micron
        del wavelengths        
        dustTable.klambda = np.copy(klambda) + Rv
        dustTable.klambda /= Rv
        del klambda
        # Initalise SCREEN class
        super(Prevot,self).__init__(Rv=Rv,dustTable=dustTable)        
        self.reference = "Prevot et al. (1984) [SMC]"             
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
    




