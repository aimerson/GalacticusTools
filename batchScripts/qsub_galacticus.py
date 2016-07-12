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
PROCS = 12
RUNS = None # e.g. "1-100"
WALLTIME = None # e.g. "08:00:00"
#######################################

compile = False
paramfile = None
clrlogs = False
iarg = 0
while iarg < len(sys.argv):
    if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
        compile = True
    if fnmatch.fnmatch(sys.argv[iarg],"-l"):
        clrlogs = True
    if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
        iarg += 1
        paramfile = sys.argv[iarg]
    iarg += 1

# If need to compile run a single job
if compile:
    RUNS = None

# Set storage directory depending on system
HOST = os.environ["HOST"]
if HOST.lower == "zodiac":
    # JPL
    storeDir = "/nobackupNFS/sunglass/"
else:
    # NERSC
    storeDir = "/global/projects/projectdir/hacc/"

# Construct logs directory
jobname = "galacticus"
logdir = storeDir+user+"/logs/galacticus/"
subprocess.call(["mkdir","-p",logdir])
logfile = logdir + jobname
if clrlogs:
    print("Clearing old log files...")
    logfiles = glob.glob(logdir+"*.OU")
    for log in logfiles:
        os.remove(log)

# Make output directory ofr array jobs
if RUNS is not None:
    version = pwd.split("/")[-1]
    outdir = storeDir+user+"/Galacticus_Out/"+version+"/"
    datestr = str(datetime.datetime.now()).split()[0].replace("-","")
    outdir = outdir + paramfile.replace(".xml","") + "/" + datestr
    subprocess.call(["mkdir","-p",outdir])
    print("OUTPUT DIRECTORY = "+outdir)    
    if not compile:
        os.system("cp Galacticus.exe "+outdir)
        if type(RUNS) == list:        
            size = len(RUNS)
    else:
        size = 0
        allruns = RUNS.split(",")
        for r in allruns:
            if "-" in r:
                minjob = int(r.split("-")[0])
                maxjob = int(r.split("-")[-1])
                size += (maxjob+1-minjob)
            else:
                size += 1
    JOB_ARRAY_SIZE = size
else:
    outdir = None
    JOB_ARRAY_SIZE = None


# Submit job
print("Submitting Galacticus job...")
args = {"compile":str(int(compile)),"paramfile":paramfile}
if outdir is not None:
    args["outdir"] = outdir
if JOB_ARRAY_SIZE is not None:
    args["JOB_ARRAY_SIZE"] = JOB_ARRAY_SIZE
submitPBS("run_galacticus.py",args=args,QUEUE=QUEUE,RUNS=RUNS,NODES=NODES,PPN=PROCS,WALLTIME=WALLTIME,\
              SHELL=SHELL,JOBNAME=jobname,LOGDIR=logdir,verbose=True,submit=True)








