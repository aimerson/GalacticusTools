#! /usr/bin/env python

import os,sys,subprocess,fnmatch,glob
import shutil
from .timing import STOPWATCH
import numpy as np


def getSubvolumes(ivols):
    """
    e.g. 1,2,3,4,5:60:3,100,102:105,500
    """
    def constructRange(value):
        elems =list(map(int,value.split(":")))
        if len(elems) == 2:
            elems = (elems[0],elems[1],1)
        return ",".join(list(map(str,range(elems[0],elems[1],elems[2]))))
    ivols = ivols.split(",")
    ranges = fnmatch.filter(ivols,"*:*")
    ivols = list(set(ivols).difference(ranges))
    ivols = ivols + [ constructRange(elem) for elem in ranges]
    ivols = list(map(int,",".join(ivols).split(",")))
    return list(np.sort(np.unique(ivols)))



class galacticusOS(object):
    
    def __init__(self,GALACTICUS_ROOT=None):
        classname = self.__class__.__name__
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        # Set location of Galacticus root
        if GALACTICUS_ROOT is None:
            self.GALACTICUS_ROOT = os.getenv("GALACTICUS_ROOT_V094")
            if self.GALACTICUS_ROOT is None:
                raise KeyError(classname+"(): Cannot locate Galacticus root directory!")
        else:
            self.GALACTICUS_ROOT = GALACTICUS_ROOT
        if not os.path.exists(self.GALACTICUS_ROOT):
            raise IOError(classname+"(): specified Galacticus root path does not exist!")
        print(classname+"(): GALACTICUS ROOT = "+self.GALACTICUS_ROOT)
        return

    
    def compile(self,clean=True,NTHREAD=1,EXE="Galacticus.exe"):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name
        print("WARNING! "+funcname+"(): compiling Galacticus...")
        sys.stdout.flush()
        SW = STOPWATCH()
        os.chdir(self.GALACTICUS_ROOT)
        sys.stdout.flush()
        # Clean build -- complete recompile
        if os.path.exists(EXE):
            os.remove(EXE)
        if clean:
            if os.path.exists("work/build/"):
                shutil.rmtree("work/build/")
        # Get number of threads to use
        NTHREAD = os.getenv("OMP_NUM_THREADS",NTHREAD)
        print(funcname+"(): Using "+str(NTHREAD)+" threads...")
        sys.stdout.flush()
        # Compile Galacticus
        cmd = "make -k -j "+str(NTHREAD)+" "+EXE        
        print(funcname+"(): Running '"+cmd+"'...")
        os.system(cmd)
        if not os.path.exists(EXE):
            SW.stop()
            raise RuntimeError(funcname+"(): Compilation failed -- unable to locate '"+EXE+"'!")
        print(funcname+"(): COMPILATION SUCCESSFUL!")
        SW.stop()
        sys.stdout.flush()
        return

    
    def run(self,paramfile,rundir=None,EXE="Galacticus.exe",NTHREAD=1,logfile=None,exitOnFail=True):
        funcname = self.__class__.__name__+"."+sys._getframe().f_code.co_name        
        # Check parameter file exists
        if not os.path.exists(paramfile):
            raise RuntimeError(funcname+"(): Parameter file '"+paramfile+"' not found!")        
        print(funcname+"(): GALACTICUS PARAMETER FILE = "+paramfile)
        # Get number of threads to use
        NTHREAD = os.getenv("OMP_NUM_THREADS",NTHREAD)
        print(funcname+"(): Using "+str(NTHREAD)+" threads...")
        sys.stdout.flush()
        # Check Galacticus compiled
        if not os.path.exists(EXE):
            self.compile(NTHREAD=NTHREAD,EXE=EXE)
        # Create run directory and move files if necessary        
        if rundir is None:
            rundir = self.GALACTICUS_ROOT
        if not os.path.exists(rundir):
            os.mkdirs(rundir)           
        if not rundir.endswith("/"):
            rundir = rundir + "/"        
        if not fnmatch.fnmatch(rundir,self.GALACTICUS_ROOT):            
            paramfileName = paramfile.split("/")[-1]
            os.rename(paramfile,rundir+paramfileName)
            if not os.path.exists(rundir+EXE):
                shutil.copyfile(self.GALACTICUS_ROOT+"/"+EXE,rundir+EXE)
        # Run Galacticus            
        os.chdir(rundir)
        print(funcname+"(): Running Galacticus...")        
        print(funcname+"(): Current directory = "+os.getcwd())
        print(funcname+"(): Running "+EXE+"...")
        SW = STOPWATCH()
        sys.stdout.flush()
        cmd = "./"+EXE + " " + paramfile 
        if logfile is not None:
            cmd = cmd + " &> "+logfile
        os.system(cmd)
        sys.stdout.flush()
        # Check logfile if specified
        if logfile is None and os.path.exists(logfile):
            pattern = "*MM: <- Finished task set*"
            failed = len(fnmatch.filter(open(logfile,'r').readlines(),pattern))==0
            if exitOnFail and failed:
                raise RuntimeError(funcname+"(): Galacticus run FAILED!")            
        print(funcname+"():Galacticus finished processing")
        SW.stop()
        sys.stdout.flush()
        return
