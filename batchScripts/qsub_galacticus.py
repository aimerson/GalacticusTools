#! /usr/bin/env python
"""
qsub_galacticus.py -- submit a Galacticus job

USAGE: ./qsub_galacticus.py -s <scratchPath> [-c] [-p <paramfile>] [-q <queue>] 
                            [-J <arrayOptions>] [-n <jobname>] [-l <resource>] 
                            [-logs] [-SUBMIT]

"""
import sys,os,getpass,fnmatch,subprocess,glob
import numpy as np
import datetime
from galacticus.utils.batchJobs import submitPBS

# If no arguments specified, print docstring and quit
if len(sys.argv) == 1:
    print(__doc__)
    quit()

USER = getpass.getuser()
pwd = subprocess.check_output(["pwd"]).replace("\n","")

# Initialise PBS class
SUBMIT = submitPBS(overwrite=True)

# Get arguments
JOBNAME = "galacticus"
COMPILE = False
PARAMS = None
rmlogs = False
SCRATCH = None
RUNS = None
QUEUE = None
SUBMIT_JOB_TO_QUEUE = False
iarg = 0
while iarg < len(sys.argv):
    if fnmatch.fnmatch(sys.argv[iarg],"-s*"):
        iarg += 1
        SCRATCH = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
        COMPILE = True
    if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
        iarg += 1
        PARAMS = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-n*"):
        iarg += 1
        JOBNAME = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-logs"):
        rmlogs = True
    if fnmatch.fnmatch(sys.argv[iarg],"-SUBMIT"):
        SUBMIT_JOB_TO_QUEUE = True
    if fnmatch.fnmatch(sys.argv[iarg],"-q"):
        iarg += 1
        QUEUE = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-J"):
        iarg += 1
        RUNS = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-l"):
        iarg += 1
        SUBMIT.addResource(sys.argv[iarg])        
    iarg += 1

# Set path to output and log directories
if SCRATCH is None:
    raise RuntimeError("Path to scratch disk not provided!")
if not SCRATCH.endswith("/"):
    SCRATCH = SCRATCH+"/"
# Create logs directory
LOGDIR = SCRATCH+"Galacticus_Logs/"
if PARAMS is not None:
    LOGDIR = LOGDIR + PARAMS.replace(".xml","") + "/"
subprocess.call(["mkdir","-p",LOGDIR])        
if rmlogs:
    for logfile in glob.glob(LOGDIR+"*"):
        os.remove(logfile)
SUBMIT.addOutputPath(LOGDIR)
SUBMIT.addErrorPath(LOGDIR)
SUBMIT.joinOutErr()

EXE = "Galacticus.exe"
ARGS = {"executable":EXE}

# Create output directory
if PARAMS is not None:
    ARGS['paramfile'] = PARAMS
    datestr = str(datetime.datetime.now()).split()[0].replace("-","")
    OUTDIR = SCRATCH+"Galacticus_Out/v0.9.4/"+PARAMS.replace(".xml","")+"/"
    ARGS['outdir'] = OUTDIR
    subprocess.call(["mkdir","-p",OUTDIR])        
    if not os.path.exists(EXE):
        COMPILE = True
    else:
        os.system("cp "+EXE+" "+OUTDIR)        
    
# If need to compile run a single job
ARGS["compile"] = int(COMPILE)
if COMPILE:
    RUNS = None

# Set remaing PBS options
SUBMIT.addQueue(QUEUE)
SUBMIT.addJobName(JOBNAME)
SUBMIT.specifyJobArray(RUNS)
ARGS["JOB_ARRAY_SIZE"] = SUBMIT.countJobs()
ARGS["JOB_MANAGER"] = SUBMIT.manager
SUBMIT.passScriptArguments(ARGS)
SUBMIT.setScript(os.path.basename(__file__).replace("qsub","run"))

# Submit job
SUBMIT.printJobString()
if SUBMIT_JOB_TO_QUEUE:
    print("Submitting Galacticus job...")
    SUBMIT.submitJob()








