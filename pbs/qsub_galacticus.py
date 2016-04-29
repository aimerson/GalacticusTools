#! /usr/bin/env python

import sys,os,getpass,fnmatch,subprocess,glob
import numpy as np


user = getpass.getuser()
pwd = subprocess.check_output(["pwd"]).replace("\n","")

#######################################
# PBS envinronment variables
SHELL = "/bin/tcsh"
QUEUE = "mediumq"  
NODES = 6
PROCS = 12
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
    
# Construct logs directory
jobname = "galacticus"
logdir = "/nobackup0/sunglass/"+user+"/logs/galacticus/"
subprocess.call(["mkdir","-p",logdir])
logfile = logdir + jobname
if clrlogs:
    logfiles = glob.glob(logdir+"*.OU")
    for log in logfiles:
        os.remove(log)

# Submit job
job = "qsub -V "
if QUEUE is not None:
    job = job + " -q " + QUEUE
if NODES is not None and PROCS is not None:
    job = job + " -l nodes=" + str(NODES) + ":ppn=" + str(PROCS)
if WALLTIME is not None:
    job = job + " -l walltime=" + str(WALLTIME)
job = job + " -S "+SHELL
#job = job + " -d " + pwd 
job = job + " -N "+ jobname + " -o " + logdir + " -j oe"
job = job + " -v compile=" + str(int(compile))+",paramfile="+str(paramfile)
job = job + " ./run_galacticus.py"
#print job
os.system(job)








