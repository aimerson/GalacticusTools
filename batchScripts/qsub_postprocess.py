#! /usr/bin/env python

import sys,os,getpass,fnmatch,subprocess,glob
import numpy as np
import datetime

from galacticus.utils.batchJobs import submitPBS

user = getpass.getuser()
pwd = subprocess.check_output(["pwd"]).replace("\n","")

#######################################
# PBS envinronment variables
SHELL = "/bin/tcsh"
QUEUE = "shortq"  
NODES = 1
PROCS = 1
RUNS = "1-9" # e.g. "1-100"
WALLTIME = None # e.g. "08:00:00"
#######################################

outdir = None
clrlogs = False
iarg = 0
while iarg < len(sys.argv):
    if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
        iarg += 1
        outdir = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-l*"):
        clrlogs = True
    iarg += 1

if outdir is None:
    raise IOError("ERROR! No output directory specified!")


# Set storage directory depending on system
HOST = os.environ["HOST"]
if HOST.lower() == "zodiac":
    # JPL
    storeDir = "/nobackupNFS/sunglass/"
else:
    # NERSC
    storeDir = "/global/projects/projectdir/hacc/"

# Construct logs directory
jobname = "postprocess"
logdir = storeDir+user+"/logs/galacticus/"
subprocess.call(["mkdir","-p",logdir])
logfile = logdir + jobname
if clrlogs:
    print("Clearing old log files...")
    logfiles = glob.glob(logdir+"*.OU")
    for log in logfiles:
        os.remove(log)

# Submit job
print("Submitting Galacticus post-processing job...")
args = {"outdir":outdir}
submitPBS("run_postprocess.py",args=args,QUEUE=QUEUE,RUNS=RUNS,NODES=NODES,PPN=PROCS,WALLTIME=WALLTIME,\
              SHELL=SHELL,JOBNAME=jobname,LOGDIR=logdir,verbose=True,submit=True)








