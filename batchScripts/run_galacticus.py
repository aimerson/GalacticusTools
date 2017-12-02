#! /usr/bin/env python
#PBS -S /bin/tcsh
"""
run_galacticus.py -- run a Galacticus job either interactively or from 
                     qsub_galacticus.py submission script.

USAGE: ./run_galacticus.py [-c] [-e <exe>] [-run] [-p <paramfile>] [-o <outdir>] [-s <scratchpath>]
                           
"""

import os,sys,subprocess,glob
import fnmatch
import datetime
import numpy as np
from galacticus.parameters import validate_parameters
from galacticus.utils.timing import STOPWATCH
from galacticus.utils.batchJobs import SLURMjob,PBSjob,NULLjob

# Get basic environment variables
USER = os.environ['USER']
GALACTICUS_ROOT = os.environ["GALACTICUS_ROOT_V094"]

# Initialise job class to extract necessary environment variables
JOB = None
if "JOB_MANAGER" in os.environ.keys():
    if os.environ['JOB_MANAGER'].upper() == "PBS":
        JOB = PBSjob(verbose=True)
    if os.environ['JOB_MANAGER'].upper() == "SLURM":
        JOB = SLURMjob(verbose=True)
else:
    if len(fnmatch.filter(os.environ.keys(),"PBS*"))>0:
        JOB = PBSjob(verbose=True)
    elif len(fnmatch.filter(os.environ.keys(),"SLURM*"))>0:
        JOB = PBSjob(verbose=True)
    else:
        JOB = None
if JOB is None:
    JOB = NULLjob(verbose=True)

# Get interactive jobs have arguments
if JOB.interactive and len(sys.argv) == 1:
    print(__doc__)
    quit()

# Import parameter options module
pwd = JOB.workDir
if pwd is None:
    pwd = subprocess.check_output(["pwd"]).replace("\n","")
sys.path.append(pwd)
from parameter_options import *

#################################################################
# INPUT ARGUMENTS
#################################################################

OUTDIR = None
SCRATCH = None
COMPILE = False
EXE = "Galacticus.exe"
RUN = False
POSTPROCESS = False
PARAM = None
JOB_ARRAY_SIZE = 1
if not JOB.interactive:
    try:
        COMPILE = bool(int(os.environ["COMPILE"]))
    except KeyError:
        pass
    try:
        PARAM = os.environ["PARAMFILE"]
    except KeyError:
        pass
    try:
        OUTDIR = os.environ["OUTDIR"]
    except KeyError:
        pass
    try:
        EXE = os.environ["EXECUTABLE"]
    except KeyError:
        pass
    try:
        JOB_ARRAY_SIZE = int(os.environ["JOB_ARRAY_SIZE"])
    except KeyError:
        pass
    try:
        RUN_GALACTICUS = bool(int(os.environ["RUN_GALACTICUS"]))
    except KeyError:
        pass
else:
    iarg = 0
    while iarg < len(sys.argv):
        if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
            COMPILE = True
        if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
            iarg += 1
            PARAM = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
            iarg += 1
            OUTDIR = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-e*"):
            iarg += 1
            EXE = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-s*"):
            iarg += 1
            SCRATCH = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-run"):
            RUN_GALACTICUS = True
        iarg += 1

#################################################################

# Compile Galacticus?
if COMPILE:
    print("-"*20)
    print("WARNING! Compiling Galacticus!")
    S = STOPWATCH()
    print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_ROOT)
    sys.stdout.flush()
    os.chdir(GALACTICUS_ROOT)
    sys.stdout.flush()
    os.system("make clean")
    os.system("make clean")
    os.system("make -k -j "+str(NTOT)+" Galacticus.exe")
    if not os.path.exists("Galacticus.exe"):
        print("*** ERROR! No Galacticus executable found!")
        sys.exit(1)
    else:
        print("COMPILATION COMPLETED SUCCESSFULLY!")
        S.stop()
    print("-"*20)
    sys.stdout.flush()

# Check parameter file specified
if PARAM is None or PARAM == "None":
    raise RuntimeError("No Galacticus parameters file provided! Use flag -p <xmlfile>.")
print("GALACTICUS PARAMETER FILE = "+PARAM)
sys.stdout.flush()

# Make output directory if not already provided
if OUTDIR is None:    
    if SCRATCH is None:
        raise RuntimeError("If no output directory specified, need to specify path to scratch space!")
    #datestr = str(datetime.datetime.now()).split()[0].replace("-","")
    OUTDIR = SCRATCH+"/Galacticus_Out/v0.9.4/"+PARAM.replace(".xml","")+"/"
    subprocess.call(["mkdir","-p",OUTDIR])
print("OUTPUT DIRECTORY = "+OUTDIR)

# Change to Galacticus root directory
os.chdir(GALACTICUS_ROOT)
# Copy executable if not found in output directory and not a job array
if not os.path.exists(OUTDIR+"/"+EXE):
    if not JOB.jobArray:
        os.system("cp Galacticus.exe "+OUTDIR+"/"+EXE)
    else:
        raise RuntimeError("Exectuable file "+EXE+" not found in "+OUTDIR)
    
# Make parameter file and copy to output directory
PARAM = OUTDIR+"/"+PARAM
if JOB.jobArray:
    PARAM = select_parameters(PARAM,jobArrayID=jobArrayID,jobArraySize=JOB_ARRAY_SIZE)
else:
    PARAM = select_parameters(PARAM,jobArrayID=-1,jobArraySize=JOB_ARRAY_SIZE)
validate_parameters(PARAM,GALACTICUS_ROOT)
os.system("mv "+PARAM+" "+OUTDIR)
print("PARAMETER FILE = "+PARAM)

# Set name of output file
if JOB.jobArray:
    ofile = "galacticus_"+str(JOB.jobArrayIndex)+".hdf5"
else:
    ofile = "galacticus.hdf5"

# Run Galacticus
if RUN_GALACTICUS:
    if "OMP_NUM_THREADS" in os.environ.keys():
        NRUN = os.environ["OMP_NUM_THREADS"]
    else:
        NRUN = str(JOB.ppn)
    # Remove old Galacticus output
    if os.path.exists(OUTDIR+"/"+ofile):
        os.remove(OUTDIR+"/"+ofile)
    # Run Galacticus
    os.chdir(OUTDIR)
    print("Current directory: "+os.getcwd())
    print("Running "+EXE+"...")
    S = STOPWATCH()
    sys.stdout.flush()
    if JOB.manager.upper() == "SLURM":
        os.system("srun -n 1 -c "+NRUN+" "+EXE+" "+PARAM)
    else:
        os.system('/usr/bin/time -f "MEMORY MAX = %M" '+EXE+" "+PARAM)
    sys.stdout.flush()
    print("GALACTICUS COMPLETE")
    S.stop()
    sys.stdout.flush()

# Post-process Galacticus run
#print("Post-processing galacticus...")
#S = STOPWATCH()
#scripts = "dustExtinguish.pl addEmLines.pl addEquivalentWidths.pl".split()
#for script in scripts:
#    path = GALACTICUS_ROOT + "/scripts/analysis/"+script
#    if os.path.exists(path):
#        os.system(path+" "+outdir+"/"+ofile)        
#S.stop()
#sys.stdout.flush()




