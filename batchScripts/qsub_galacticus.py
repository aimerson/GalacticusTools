#! /usr/bin/env python

import sys,os,getpass,fnmatch,subprocess,glob
import numpy as np
import datetime

user = getpass.getuser()
pwd = subprocess.check_output(["pwd"]).replace("\n","")

#######################################
# PBS envinronment variables
SHELL = "/bin/tcsh"
QUEUE = "longq"  
NODES = 1
PROCS = 12
RUNS = "1-2"
WALLTIME = None
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

# Construct logs directory
jobname = "galacticus"
logdir = "/nobackup0/sunglass/"+user+"/logs/galacticus/"
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
    outdir = "/nobackup0/sunglass/"+user+"/Galacticus_Out/"+version+"/"
    datestr = str(datetime.datetime.now()).split()[0].replace("-","")
    outdir = outdir + paramfile.replace(".xml","") + "/" + datestr
    subprocess.call(["mkdir","-p",outdir])
    print("OUTPUT DIRECTORY = "+outdir)    
    if not compile:
        os.system("cp galacticus.exe "+outdir)
else:
    outdir = None

# Submit job
print("Submitting Galacticus job...")
job = "qsub -V"
if QUEUE is not None:
    job = job + " -q " + QUEUE
if RUNS is not None:
    if type(RUNS) is list:
        RUNS = ",".join(map(str,RUNS))
    job = job + " -J " + str(RUNS)
if NODES is not None and PROCS is not None:
    job = job + " -l nodes=" + str(NODES) + ":ppn=" + str(PROCS)
if WALLTIME is not None:
    job = job + " -l walltime=" + str(WALLTIME)
job = job + " -S "+SHELL
#job = job + " -d " + pwd 
job = job + " -N "+ jobname + " -o " + logdir + " -j oe"
job = job + " -v compile=" + str(int(compile))+",paramfile="+str(paramfile)
if outdir is not None:
    job = job + ",outdir="+outdir
if RUNS is not None:
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
    job = job + ",JOB_ARRAY_SIZE="+str(size)
job = job + " ./run_galacticus.py"
print job
os.system(job)








