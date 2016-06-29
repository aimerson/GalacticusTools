#! /usr/bin/env python
#PBS -S /bin/tcsh

import os,sys,subprocess,getpass,glob
import fnmatch
import datetime
import numpy as np
user = getpass.getuser()

try:
    pwd = os.environ["PBS_O_WORKDIR"]
except KeyError:
    pwd = subprocess.check_output(["pwd"]).replace("\n","")   

sys.path.append(pwd)
from parameter_options import *
from galacticus.parameters import validate_parameters
from galacticus.utils.timing import STOPWATCH


#################################################################
# BATCH JOB VARIABLES
#################################################################

# i) job name
try:
    jobID = os.environ["PBS_JOBNAME"]
    print("JOB NAME = "+jobID)
    batchjob = True
    interactive = False
except KeyError:
    jobID = None
    batchjob = False
    interactive = True
# ii) job array ID
try:
    jobArrayID = int(os.environ["PBS_ARRAY_INDEX"])
    print("JOB INDEX = "+str(jobArrayID))
except KeyError:
    jobArrayID = -1
# iii) Job environment
if batchjob:
    environment = os.environ["PBS_ENVIRONMENT"]
    print("JOB ENVIRONMENT = "+str(environment))
    if fnmatch.fnmatch(environment,"*_INTERACTIVE"):
        interactive = True
sys.stdout.flush()

# Get number of nodes
if batchjob:
    nodefile = os.environ["PBS_NODEFILE"]
    nodes = np.loadtxt(nodefile,usecols=[0],dtype=str)
    NPROC = nodes.size
    print("Using "+str(NPROC)+" processors...")
    sys.stdout.flush()


#################################################################
# INPUT ARGUMENTS
#################################################################

outdir = None
compile = False
PARAM = None
JOB_ARRAY_SIZE = 1
if not interactive:
    try:
        compile = bool(int(os.environ["compile"]))
    except KeyError:
        pass
    try:
        PARAM = os.environ["paramfile"]
    except KeyError:
        pass
    try:
        outdir = os.environ["outdir"]
    except KeyError:
        pass
    try:
        JOB_ARRAY_SIZE = int(os.environ["JOB_ARRAY_SIZE"])
    except KeyError:
        pass
else:
    iarg = 0
    while iarg < len(sys.argv):
        if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
            compile = True
        if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
            iarg += 1
            PARAM = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
            iarg += 1
            outdir = sys.argv[iarg]
        iarg += 1

#################################################################

# Compile Galacticus?
EXE = "Galacticus.exe"    
GALACTICUS_ROOT = os.environ["GALACTICUS_ROOT_V094"]
if compile:
    print("WARNING! Compiling Galacticus!")
    S = STOPWATCH()
    print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_ROOT)
    sys.stdout.flush()
    os.chdir(GALACTICUS_ROOT)
    sys.stdout.flush()
    os.system("make clean")
    os.system("make clean")
    os.system("make "+EXE)
    if not os.path.exists(EXE):
        print("*** ERROR! No Galacticus executable found!")
        sys.exit(1)
    else:
        print("COMPILATION COMPLETED SUCCESSFULLY!")
        S.stop()

# Check parameter file specified
if PARAM is None or PARAM == "None":
    print("*** ERROR! No Galacticus parameters file provided!")
    print("           Use flag -p <xmlfile>")
    sys.exit(1)
else:    
    print("GALACTICUS PARAMETER FILE = "+PARAM)
    sys.stdout.flush()

# Set Galacticus version
version = pwd.split("/")[-1]

# Make output directory if not already provided
if outdir is None:
    outdir = "/nobackup0/sunglass/"+user+"/Galacticus_Out/"+version+"/"
    datestr = str(datetime.datetime.now()).split()[0].replace("-","")
    outdir = outdir + PARAM.replace(".xml","") + "/" + datestr
    subprocess.call(["mkdir","-p",outdir])
print("OUTPUT DIRECTORY = "+outdir)

# Copy Galacticus executable (single jobs only)
os.chdir(GALACTICUS_ROOT)
if jobArrayID < 0:
    os.system("cp "+EXE+" "+outdir)

# Make parameter file and copy to output directory
PARAM = select_parameters(PARAM,jobArrayID=jobArrayID,jobArraySize=JOB_ARRAY_SIZE)
validate_parameters(PARAM,GALACTICUS_ROOT)
os.system("mv "+PARAM+" "+outdir)
print("PARAMETER FILE = "+PARAM)

# Remove old Galacticus output
if jobArrayID < 0:
    ofile = "galacticus.hdf5"
else:
    ofile = "galacticus_"+str(jobArrayID)+".hdf5"
if os.path.exists(outdir+"/"+ofile):
    os.remove(outdir+"/"+ofile)


# Run Galacticus
os.chdir(outdir)
print("Running "+EXE+"...")
S = STOPWATCH()
sys.stdout.flush()
os.system(EXE+" "+PARAM)
sys.stdout.flush()
print("COMPLETE (run_galacticus.py)")
S.stop()
sys.stdout.flush()
