#! /usr/bin/env python

import sys,os
import fnmatch
import numpy as np
from scipy.integrate import romberg,quad
import scipy.special as spspec


######################################################################################
# Schechter functional form of LF
######################################################################################

class SchechterMagnitudes(object):

    def __init__(self,alpha,Mstar,phistar):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.alpha = alpha
        self.Mstar = Mstar
        self.phistar = phistar
        return

    def phi(self,mags):
        """
        phi(): Returnn Schechter luminosity function for specified
               absolute magnitude(s)

        USAGE: lf = SchechterFunction.phi(mags)

        """
        f = 10.0**(0.4*(self.Mstar-mags))
        lf = 0.4*np.log(10.0)*self.phistar*(f**(self.alpha+1.0))
        lf = lf/np.exp(f)
        return lf

    def integrate(self,low,upp):
        """
        integrate(): Integrate Schecter luminosity function between
                        specified absolute magnitude limits

        USAGE: integral = ShcechterFunction.integrate(low,upp)
        
        """
        return romberg(self.phi,mag_low,mag_upp)


    def numberBrighter(self,Msol=None,L_low=None):
        """
        numberBrighter(): Integrate Schechter function to return
                            number of galaxies brighter than L=L_low.

        USAGE: SchechterFunction.numberBrighter([Msol],[L_low])
  
               Msol = absolute magnitude of the sun (necessary if
                       L_low != 0) 
               L_low = limiting luminosity (default L_low = 0)

        """

        if L_low is None:
            return self.phistar*spspec.gamma(self.alpha+1.0)
        else:
            Lstar = 10.0**((Msol-self.Mstar)/2.5)
            return self.phistar*spspec.gammainc(self.alpha+1.0,L_low/Lstar)

    def luminosityBrighter(self,Msol,L_low=None):
        """                                                                                                                                                                                                                 
        luminosityBrighter(): Integrate Schechter function to return
                                total luminosity of galaxies brighter
                                than L=L_low.
                                                                                                                                                                                                                            
        USAGE: SchechterFunction.luminosityBrighter(Msol,[L_low])

               Msol  = absolute magnitude of the sun
               L_low = limiting luminosity (default L_low = 0)                                                                                                                                                              
        """
        import scipy.special as spspec
        Lstar = 10.0**((Msol-self.Mstar)/2.5)
        if L_low is None:
            return self.phistar*Lstar*spspec.gamma(self.alpha+2.0)
        else:
            return self.phistar*Lstar*spspec.gammainc(self.alpha+2.0,L_low/Lstar)


class SchechterLuminosities(object):

    def __init__(self,alpha,Lstar,phistar):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.alpha = alpha
        self.Lstar = Lstar
        self.phistar = phistar
        return

    def phi(self,lum,perDex=True):
        """
        phi(): Return Schechter luminosity function for specified
                               luminosities

        USAGE: lf = SchechterFunction.phi(lum,[perDex])

        """
        factor1 = (lum/self.Lstar)**self.alpha
        factor2 = np.exp(-lum/self.Lstar)
        lf = self.phistar*factor1*factor2
        if perDex:
            lf *= np.log(10.0)*(lum/self.Lstar)
        return lf


    def numberBrighter(self,L_low=None):
        """
        numberBrighter(): Integrate Schechter function to return
                            number of galaxies brighter than L=L_low.

        USAGE: SchechterFunction.numberBrighter([L_low])
  
               L_low = limiting luminosity (default L_low = 0)

        """

        if L_low is None:
            return self.phistar*spspec.gamma(self.alpha+1.0)
        else:
            return self.phistar*spspec.gammainc(self.alpha+1.0,L_low/self.Lstar)

    def luminosityBrighter(self,L_low=None):
        """                                                                                                                                                                                                                 
        luminosityBrighter(): Integrate Schechter function to return
                                total luminosity of galaxies brighter
                                than L=L_low.
                                                                                                                                                                                                                            
        USAGE: SchechterFunction.luminosityBrighter([L_low])

               L_low = limiting luminosity (default L_low = 0)                                                                                                                                                              
        """
        import scipy.special as spspec
        if L_low is None:
            return self.phistar*self.Lstar*spspec.gamma(self.alpha+2.0)
        else:
            return self.phistar*self.Lstar*spspec.gammainc(self.alpha+2.0,L_low/self.Lstar)


######################################################################################
# Saunders functional form for luminosity function
######################################################################################

class SaundersLuminosities(object):

    def __init__(self,Lstar,C,alpha,sigma):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        self.Lstar = Lstar
        self.C = C
        self.alpha = alpha
        self.sigma = sigma
        return

    def phi(self,lum,perDex=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        factor1 = (lum/self.Lstar)**self.alpha
        factor2 = -1.0/(2.0*self.sigma**2)
        factor3 = np.log10(1.0+(lum/self.Lstar))
        lf = self.C*factor1*np.exp(factor2*factor3**2)
        if perDex:
            lf *= np.log(10.0)*(lum/self.Lstar)
        return lf



######################################################################################
# EUCLID analytic fits to LF
######################################################################################

class EuclidModel1(object):

    def __init__(self,alpha=-1.35,lstar0=10.**41.5,phistar0=10**-2.8,delta=2.0,epsilon=1.0,zbreak=1.3):
        self.number = 1
        self.name = "Pozzetti"
        self.alpha = alpha
        self.lstar0 = lstar0
        self.phistar0 = phistar0
        self.delta = delta
        self.epsilon = epsilon
        self.zbreak = zbreak
        return

    def phistar(self,z):
        if z <= self.zbreak:
            phistar = self.phistar0*((1.0+z)**self.epsilon)
        else:
            phistar = self.phistar0*((1.0+self.zbreak)**(2.0*self.epsilon))*((1.0+z)**(-1.0*self.epsilon))        
        return phistar

    def Lstar(self,z):
        return self.lstar0*(1.0+z)**self.delta

    def phi(self,l,z,perDex=True):
        # Luminosity function at a redshift, z
        lstar = self.Lstar(z)        
        factor1 = (l/lstar)**self.alpha
        factor2 = np.exp(-l/lstar)
        lf = self.phistar(z)*factor1*factor2/lstar
        if perDex:
            lf *= np.log(10.0)*l
        return lf
    
class EuclidModel2(object):

    def __init__(self,alpha=-1.4,lstarbreak=10**42.59,phistar0=10**-2.7,c=0.22,epsilon=0.0,zbreak=2.23,Az=1.0):
        self.number = 2
        self.name = "Geach"
        self.alpha = alpha
        self.phistar0 = phistar0
        self.epsilon = epsilon
        self.zbreak = zbreak
        self.az = Az
        self.c = c
        self.lstarbreak = lstarbreak
        return

    def Lstar(self,z):
        lstar = -1.0*self.c*((z-self.zbreak)**2) + np.log10(self.lstarbreak)
        lstar = 10**lstar
        return lstar

    def phistar(self,z):
        if z <= self.zbreak:
            phistar = self.phistar0*((1.0+z)**self.epsilon)
        else:
            phistar = self.phistar0*((1.0+self.zbreak)**(2.0*self.epsilon))*((1.0+z)**(-1.0*self.epsilon))        
        return phistar

    def phi(self,l,z,perDex=True):
        # Luminosity function at a redshift, z
        lstar = self.Lstar(z)        
        factor1 = (l/lstar)**self.alpha
        factor2 = np.exp(-l/lstar)
        phistar = self.phistar(z)
        lf = phistar*factor1*factor2/lstar
        if perDex:
            lf *= np.log(10.0)*l
        return lf


class EuclidModel3(object):

    def __init__(self,alpha=-1.587,beta=1.615,gamma=1.0,delta=2.288,phistar0=10.0**-2.920,lstarinf=10**42.956,\
                     lstarhalf=10**41.733,method="broken"):
        self.number = 3
        self.name = "Hirata"        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.phistar0 = phistar0
        self.lstarinf = lstarinf
        self.lstarhalf = lstarhalf
        if fnmatch.fnmatch(method,"*broken*") or fnmatch.fnmatch(method,"*power*"):
            self.method = "broken"
            self.gamma = 1.0
        elif method.lower() == "hybrid":
            self.method = "hybrid"
            self.delta = 2.0
        else:
            self.method = "other"        
        return


    def Lstar(self,z):
        logLstar = np.log10(self.lstarinf) + ((1.5/(1.0+z))**self.beta)*np.log10(self.lstarhalf/self.lstarinf)
        return 10**logLstar
        
    def phistar(self,z):
        return self.phistar0

    def phi(self,l,z,perDex=True):
        # Luminosity function at a redshift, z
        Lstar = self.Lstar(z)
        factor1 = (l/Lstar)**self.alpha
        factor2 = np.exp(-(1.0-self.gamma)*l/Lstar)
        factor3 = (1.0 + np.expm1(1.0)*((l/Lstar)**self.delta))**-self.gamma
        lf = factor1*factor2*factor3*self.phistar(z)/Lstar
        if perDex:
            lf *= np.log(10)*l
        return lf



class EuclidLuminosityFunction(object):
    
    def __init__(self,model,**kwargs):
        self.number = model        
        if self.number == 1:
            self.model = EuclidModel1(**kwargs)
        elif self.number == 2:
            self.model = EuclidModel2(**kwargs)
        else:
            self.model = EuclidModel3(**kwargs)
        return

    def Lstar(self,z):
        return self.model.Lstar(z)
    
    def phistar(self,z):
        return self.model.phistar(z)

    def phi(self,luminosity,z,perDex=True):
        return self.model.phi(luminosity,z,perDex=perDex)

    def integrate(self,llow,lupp,z,**kwargs):
        kwargs["args"] = (z,False)
        integral = quad(self.phi,llow,lupp,**kwargs)
        return integral






