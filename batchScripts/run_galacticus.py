#! /usr/bin/env python
#PBS -S /bin/tcsh

import os,sys,subprocess,getpass,glob
import fnmatch
import datetime
import numpy as np
user = getpass.getuser()


from galacticus.parameters import validate_parameters
from galacticus.utils.timing import STOPWATCH
from galacticus.utils.batchJobs import SLURMjob,PBSjob

print("#########################################")
print("#         GALACTICUS LOG FILE           #")
print("#########################################")
print("START TIME = "+str(datetime.datetime.now()))
print("----------------------------------------")
print("Batch job variables:")
if len(fnmatch.filter(os.environ.keys(),"PBS*"))>0:
    JOB = PBSjob(verbose=True)
else:
    JOB = SLURMjob(verbose=True)
print("----------------------------------------")
pwd = JOB.workDir
if pwd is None:
    pwd = subprocess.check_output(["pwd"]).replace("\n","")
sys.path.append(pwd)
from parameter_options import *

#################################################################
# BATCH JOB VARIABLES
#################################################################

if JOB.jobName is None:
    batchjob = False
jobArrayID = JOB.taskID
if jobArrayID is None:
    jobArrayID = -1
if JOB.cpus is None:
    NPROC = 1
    NTOT = 1
else:
    NPROC = JOB.ppn
    NTOT = JOB.cpus
    print("Number of nodes = "+str(JOB.nodes))
    print("Processors per nodes = "+str(JOB.ppn))
    print("Total number of CPUS = "+str(JOB.cpus))
sys.stdout.flush()

#################################################################
# INPUT ARGUMENTS
#################################################################

outdir = None
compile = False
PARAM = None
JOB_ARRAY_SIZE = 1
if not JOB.interactive:
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
    print("----------------------------------------")
    print("WARNING! Compiling Galacticus!")
    S = STOPWATCH()
    print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_ROOT)
    sys.stdout.flush()
    os.chdir(GALACTICUS_ROOT)
    sys.stdout.flush()
    os.system("make clean")
    os.system("make clean")
    os.system("make -k -j "+str(NTOT)+" "+EXE)
    if not os.path.exists(EXE):
        print("*** ERROR! No Galacticus executable found!")
        sys.exit(1)
    else:
        print("COMPILATION COMPLETED SUCCESSFULLY!")
        S.stop()
    print("----------------------------------------")
    sys.stdout.flush()

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
    outdir = "/nobackupNFS/sunglass/"+user+"/Galacticus_Out/"+version+"/"
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
if JOB.manager.upper() == "SLURM":
    os.system("srun -n 1 -c "+NPROC+" "+EXE+" "+PARAM)
else:
    os.system(EXE+" "+PARAM)
sys.stdout.flush()
print("COMPLETE (run_galacticus.py)")
S.stop()
sys.stdout.flush()
