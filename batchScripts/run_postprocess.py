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
print("#  GALACTICUS POST-PROCESSING LOG FILE  #")
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
force = False
if not JOB.interactive:
    try:
        outdir = os.environ["outdir"]
    except KeyError:
        pass
    try:
        force = bool(int(os.environ["force_overwrite"]))
    except KeyError:
        pass
else:
    iarg = 0
    while iarg < len(sys.argv):
        if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
            iarg += 1
            outdir = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-f*"):
            force = True
        iarg += 1

if outdir is None:
    raise IOError("ERROR! No output directory specified!")
sys.stdout.flush()    

#################################################################

# Get files to post-process
files = glob.glob(outdir+"/galacticus_"+str(jobArrayID)+"*.hdf5")
print("Selected following files:")
space = "\n     "
print(space+space.join(files))
sys.stdout.flush()    

# Set merge output file
ofile = outdir + "/galacticus_merge"+str(jobArrayID)+".hdf5"
print("OUTPUT FILE = "+ofile)
if force:
    os.remove(ofile)
sys.stdout.flush()    

# Get Galacticus root path
GALACTICUS_ROOT = os.environ["GALACTICUS_ROOT_V094"]
print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_ROOT)
sys.stdout.flush()    

# Merge output files
if not os.path.exists(ofile):
    print("Merging files...")
    sys.stdout.flush()
    S = STOPWATCH()
    cmd = GALACTICUS_ROOT+"//scripts/aux/Merge_Models.pl "+" ".join(files)+" "+ofile
    os.system(cmd)
    print("MERGING COMPLETED SUCCESSFULLY!")
    S.stop()
    sys.stdout.flush()

# Process merged file
print("Post-processing merged file...")
sys.stdout.flush()
S = STOPWATCH()
# i) Dust
cmd = GALACTICUS_ROOT+"//scripts/aanalysis/dustExtinguish.pl"+" "+ofile
os.system(cmd)
sys.stdout.flush()
# ii) Emission lines
cmd = GALACTICUS_ROOT+"//scripts/aanalysis/addEmLines.pl"+" "+ofile
os.system(cmd)
sys.stdout.flush()
# iii) Equivalent widths
cmd = GALACTICUS_ROOT+"//scripts/aanalysis/addEquivalentWidths.pl"+" "+ofile
os.system(cmd)
sys.stdout.flush()
print("POST-PROCESSING COMPLETED SUCCESSFULLY!")
S.stop()
sys.stdout.flush()

