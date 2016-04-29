#! /usr/bin/env python

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
    chunk = int(os.environ["PBS_ARRAYID"]) - 1
    print("JOB INDEX = "+str(chunk))
except KeyError:
    chunk = -1
# iii) Job environment
if batchjob:
    environment = os.environ["PBS_ENVIRONMENT"]
    print("JOB ENVIRONMENT = "+str(environment))
    if fnmatch.fnmatch(environment,"*_INTERACTIVE"):
        interactive = True
sys.stdout.flush()

#################################################################
# INPUT ARGUMENTS
#################################################################

compile = False
PARAM = None
if not interactive:
    try:
        compile = bool(int(os.environ["compile"]))
    except KeyError:
        pass
    try:
        PARAM = os.environ["paramfile"]
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
        iarg += 1

if PARAM is None:
    print("*** ERROR! No Galacticus parameters file provided!")
    print("           Use flag -p <xmlfile>")
    sys.exit(1)
else:    
    print("GALACTICUS PARAMETER FILE = "+PARAM)
    sys.stdout.flush()
#################################################################

if batchjob:
    nodefile = os.environ["PBS_NODEFILE"]
    nodes = np.loadtxt(nodefile,usecols=[0],dtype=str)
    NPROC = nodes.size
    print("Using "+str(NPROC)+" processors...")
    sys.stdout.flush()


# Set Galacticus version
version = pwd.split("/")[-1]

# Make output directory
outdir = "/nobackup0/sunglass/"+user+"/Galacticus_Out/"+version+"/"
datestr = str(datetime.datetime.now()).split()[0].replace("-","")
outdir = outdir + datestr
n = len(glob.glob(outdir+"-*"))
outdir = outdir + "-" + str(n+1).zfill(3)
subprocess.call(["mkdir","-p",outdir])

# Compile Galacticus?
EXE = "Galacticus.exe"    
GALACTICUS_ROOT = os.environ["GALACTICUS_ROOT_V094"]
if compile:
    print("WARNING! Compiling Galacticus!")
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

# Make parameter file and copy to output directory
os.chdir(GALACTICUS_ROOT)
os.system("cp "+EXE+" "+outdir)
select_parameters(PARAM)
validate_parameters(PARAM,GALACTICUS_ROOT)
os.system("cp "+PARAM+" "+outdir)

# Run Galacticus
os.chdir(outdir)
print("Running "+EXE+"...")
sys.stdout.flush()
os.system(EXE+" "+PARAM)
sys.stdout.flush()
print("COMPLETE (run_galacticus.py)")
sys.stdout.flush()
