#! /usr/bin/env python

import numpy as np
from scipy.integrate import romberg,quad
import scipy.special as spspec


######################################################################################
# Schechter functional form of LF
######################################################################################

class SchechterFunction(object):

    def __init__(self,alpha,Mstar,phistar):
        self.alpha = alpha
        self.Mstar = Mstar
        self.phistar = phistar

    def phi(self,mags):
        """
        phi(): Return Schechter luminosity function for                                                                                                                                                     
                               specified absolute magnitude(s)

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
        


######################################################################################
# EUCLID analytic fits to LF
######################################################################################

class EuclidModel1(object):

    def __init__(self,alpha=-1.35,lstar0=10.**41.5,phistar0=10**-2.,delta=2.0,epsilon=1.0,zbreak=1.3):
        self.number = 1
        self.name = "Pozzetti"
        self.alpha = alpha
        self.lstar0 = lstar0
        self.phistar0 = phistar0
        self.delta = delta
        self.epsilon = epsilon
        self.zbreak = zbreak
        return

    def phi(self,l,z):
        # Luminosity function at a redshift, z
        lstar = self.lstar0*(1.0+z)**self.delta
        if z <= self.zbreak:
            phistar = self.phistar0*((1.0+z)**self.epsilon)
        else:
            phistar = self.phistar0*((1.0+self.zbreak)**(2.0*self.epsilopn))*((1.0+z)**(-1.0*self.epsilon))
        lf = (phistar/lstar)*((l/lstar)**self.alpha)*np.exp(-l/lstar)
        return lf
    
class EuclidModel2(object):

    def __init__(self,alpha=-1.4,lstarbreak=10**42.59,phistar0=10**-2.7,c=0.22,epsilon=0.0,zbreak=2.23,Az=1.0):
        self.number = 2
        self.name = "Geach"
        self.alpha = alpha
        self.phistar = phistar0
        self.delta = delta
        self.epsilon = epsilon
        self.zbreak = zbreak
        self.az = Az
        self.c = c
        self.lstarbreak = lstarbreak
        return

    def phi(self,l,z):
        # Luminosity function at a redshift, z
        lstar = -1.0*self.c*((z-self.zbreak)**2) + np.log10(self.lstarbreak)
        lstar = 10**lstar
        if z <= self.zbreak:
            phistar = self.phistar0*((1.0+z)**self.epsilon)
        else:
            phistar = self.az*((1.0+z)**(-1.0*self.epsilon))
        lf = (phistar/lstar)*((l/lstar)**alpha)*np.exp(-l/lstar)
        return lf


class EuclidModel3(object):

    def __init__(self,alpha=-1.45,beta=31.8,gamma=0.86,phistar=10.0**-2.62,lstarinf=10**42.44,lstarhalf=10**41.56):
        self.number = 3
        self.name = "Hirata"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.phistar = phistar
        self.lstarinf = lstarinf
        self.lstarhalf = lstarhalf
        return

    def phi(self,l,z):
        # Luminosity function at a redshift, z
        logLstar = np.log10(self.lstarinf) + ((1.5/(1.0+z))**self.beta)*np.log10(self.lstarhalf/self.lstarinf)
        Lstar = 10**logLstar
        lf = ((1.0+np.expm1(1.0)*((l/Lstar)**2))**-self.gamma)/Lstar
        lf *= np.exp(-(1-self.gamma)*(l/Lstar))*self.phistar*((l/Lstar)**self.alpha)
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

    def phi(self,luminosity,z):
        return self.model.phi(luminosity,z)

    def integrate(self,llow,lupp,z,**kwargs):
        kwargs["args"] = (z)
        integral = integrate.quad(self.phi,llow,lupp,**kwargs)
        return integral






