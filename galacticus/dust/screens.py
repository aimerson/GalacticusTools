#! /usr/bin/env python

# Values/fits from: http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html

import sys,re
import numpy as np
import copy
import fnmatch
from scipy.interpolate import interp1d
from .utils import DustProperties
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

    def attenuation(self,wavelength,Av):        
        return self.dustCurve(wavelength*angstrom/micron)*Av



class dustScreen(DustProperties):
    
    def __init__(self,verbose=False):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Set verbosity
        self._verbose = verbose
        # Initialise DustProperties class
        super(dustScreen, self).__init__()
        # Dictionary to store classes for different screens
        self.screenObjs = {}
        return

    def attenuate(self,galHDF5Obj,z,datasetName,overwrite=False,returnDataset=True,progressObj=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Check if dust attenuated luminosity already calculated
        if datasetName in galHDF5Obj.availableDatasets(z) and not overwrite:
            if progressObj is not None:
                progressObj.increment()
                progressObj.print_status_line()
            if returnDataset:
                out = galHDF5Obj.selectOutput(z)
                return np.array(out["nodeData/"+datasetName])
            else:
                return
       # Check if a total luminosity or disk/spheroid luminosity                                                                                                         
        if datasetName.startswith("total"):
            diskResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","disk"),overwrite=False,returnDataset=True)
            spheroidResult = self.attenuate(galHDF5Obj,z,datasetName.replace("total","spheroid"),overwrite=False,returnDataset=True)
            result = np.copy(diskResult) + np.copy(spheroidResult)
            del diskResult,spheroidResult
        else:
            result = self.computeAttenuation(galHDF5Obj,z,datasetName)
        # Write property to file and return result
        out = galHDF5Obj.selectOutput(z)
        galHDF5Obj.addDataset(out.name+"/nodeData/",datasetName,result)
        attr = None
        if fnmatch.fnmatch(datasetName,"*LuminositiesStellar*"):
            attr = {"unitsInSI":luminosityAB}
        if fnmatch.fnmatch(datasetName,"*LineLuminosity*"):
            attr = {"unitsInSI":luminositySolar}
        if attr is not None:
            galHDF5Obj.addAttributes(out.name+"/nodeData/"+datasetName,attr)
        if progressObj is not None:
            progressObj.increment()
            progressObj.print_status_line()
        if returnDataset:
            return result
        return

    
    def selectScreen(self,screenName,Av,age=None,Rv=None):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Store and return appropriate screen class
        if screenName not in self.screenObjs.keys():
            if fnmatch.fnmatch(screenName,"calzetti*"):                
                self.screenObjs[screenName] = Calzetti(Rv=Rv)
            if fnmatch.fnmatch(screenName,"allen*"):                
                self.screenObjs[screenName] = Allen(Rv=Rv)
            if fnmatch.fnmatch(screenName,"fitzpatrick*"):                
                self.screenObjs[screenName] = Fitzpatrick(Rv=Rv,galaxy="LMC")
            if fnmatch.fnmatch(screenName,"seaton*"):                
                self.screenObjs[screenName] = Fitzpatrick(Rv=Rv,galaxy="MW")
            if fnmatch.fnmatch(screenName,"prevot*"):                
                self.screenObjs[screenName] = Prevot(Rv=Rv)
        return self.screenObjs[screenName]
    

    def computeAttenuation(self,galHDF5Obj,z,datasetName):        
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Get nearest redshift output
        out = galHDF5Obj.selectOutput(z)
        # Compute attenuation
        if self._verbose:
            print(funcname+"(): Processing dataset '"+datasetName+"'")
        # Check is a luminosity for attenuation
        MATCH = re.search(r"^(disk|spheroid)(LuminositiesStellar|LineLuminosity):([^:]+):([^:]+):z([\d\.]+)(:contam_[^:]+)?:dustScreen_([^:]+)",datasetName)
        if not MATCH:
            raise ParseError(funcname+"(): Cannot parse '"+datasetName+"'!")
       # Extract dataset information                                                                                                                                     
        component = MATCH.group(1)
        if component not in ["disk","spheroid"]:
            raise ParseError(funcname+"(): Cannot identify if '"+datasetName+"' corresponds to disk or spheroid!")
        luminosityType = MATCH.group(2)
        filter = MATCH.group(3)
        frame = MATCH.group(4)
        redshift = MATCH.group(5)
        if self._verbose:
            infoLine = "filter={0:s}  frame={1:s}  redshift={2:s}".format(filter,frame,redshift)
            print(funcname+"(): Filter information:\n        "+infoLine)
        contamination = MATCH.group(6)
        if contamination is None:
            contamination = ""
        screenOptions = MATCH.group(7)
        dustExtension = ":dustScreen_"+screenOptions
        # Check dust screen information is as expected
        OPTIONS = re.search(r"^([^_]+)(_age[\d\.]+)?_Av([\d\.]+)",screenOptions)
        if not OPTIONS:
            raise ParseError(funcname+"(): Dust screen options '"+screenOptions+"' not understood!")
        screenName = OPTIONS.group(1).lower()
        age = OPTIONS.group(2).replace("_age","")        
        if age == "":
            age = None
        else:
            age = float(age)
        Av = float(OPTIONS.group(3))
        # Get name of unattenuated dataset
        luminosityDataset = datasetName.replace(dustExtension,"")
        # Construct filter label
        filterLabel = filter+":"+frame+":z"+redshift+contamination
        # Compute effective wavelength for filter/line
        if frame == "observed":
            effectiveWavelength = self.effectiveWavelength(filter,redshift=float(redshift),verbose=self._verbose)
        else:
            effectiveWavelength = self.effectiveWavelength(filter,redshift=None,verbose=self._verbose)
        # Select class for dust screen and compute attenuation in magnitudes
        screenClass = self.selectScreen(screenName,Av,age=age,Rv=None)
        attenuation = screenClass.attenuation(effectiveWavelength,Av)
        # Convert to optical depth
        e = np.exp(1.0)
        opticalDepth = np.copy(attenuation)/(2.5*np.log10(e))
        attenuation = np.exp(-opticalDepth)
        # Apply attenuation and return result                                                                                                                            
        result = np.array(out["nodeData/"+luminosityDataset])*attenuation
        return result



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
    




