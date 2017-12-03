#! /usr/bin/env python
"""
qsub_getMillenniumMergerTrees.py -- submit a job array to build Millennium merger trees

USAGE: ./qsub_getMillenniumMergerTrees.py -s <scratchPath> [-c] [-e <exe>] [-sim <simulation>] 
                                          [-u <username>] [-p <password>] [-all] [-force] [-logs]
                                          [-q <queue>] [-J <arrayOptions>] [-n <jobname>] 
                                          [-l <resource>] [-SUBMIT] 

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

#################################################################
# INPUT ARGUMENTS
#################################################################

EXE = "Millennium_Merger_Tree_File_Maker.exe"
COMPILE = False
SIMULATION = None
OVERWRITE = False
USERNAME = None
PASSWORD = None
SCRATCH = None
PROCESS_ALL_SUBVOLUMES = False
JOBNAME = "millenniumTrees"
rmlogs = False
RUNS = "1-512"
QUEUE = None
SUBMIT_JOB_TO_QUEUE = False
iarg = 0
while iarg < len(sys.argv):
    if fnmatch.fnmatch(sys.argv[iarg],"-s"):
        iarg += 1
        SCRATCH = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-all"):
        PROCESS_ALL_SUBVOLUMES = True
    if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
        COMPILE = True
    if fnmatch.fnmatch(sys.argv[iarg],"-sim"):
        iarg += 1
        SIMULATION = sys.argv[iarg]        
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
    if fnmatch.fnmatch(sys.argv[iarg],"-force"):
        OVERWRITE = True
    if fnmatch.fnmatch(sys.argv[iarg],"-u*"):
        iarg += 1
        USERNAME = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
        iarg += 1
        PASSWORD = sys.argv[iarg]
    if fnmatch.fnmatch(sys.argv[iarg],"-e*"):
        iarg += 1
        EXE = sys.argv[iarg]
    iarg += 1

#################################################################

ARGS = {}

# Check username and password provided
if USERNAME is None:
    if "MPA_MILLENNIUM_USERNAME" in os.environ.keys():
        USERNAME = os.environ["MPA_MILLENNIUM_USERNAME"]
    else:
        raise RuntimeError("Username for Millennium Database not specified!")
if PASSWORD is None:
    if "MPA_MILLENNIUM_PASSWORD" in os.environ.keys():
        PASSWORD = os.environ["MPA_MILLENNIUM_PASSWORD"]
    else:
        raise RuntimeError("Password for Millennium Database not specified!")
ARGS["USERNAME"] = USERNAME
ARGS["PASSWORD"] = PASSWORD

# Set path to output and log directories
if SCRATCH is None:
    raise RuntimeError("Path to scratch disk not provided!")
if not SCRATCH.endswith("/"):
    SCRATCH = SCRATCH+"/"
# Create logs directory
LOGDIR = SCRATCH+"Galacticus_Logs/mergerTrees/"
subprocess.call(["mkdir","-p",LOGDIR])        
if rmlogs:
    for logfile in glob.glob(LOGDIR+"*"):
        os.remove(logfile)
SUBMIT.addOutputPath(LOGDIR)
SUBMIT.addErrorPath(LOGDIR)
SUBMIT.joinOutErr()

# Store arguments to pass to job
ARGS["EXECUTABLE"] = EXE
ARGS["OVERWRITE"] = int(OVERWRITE)
    
# If need to compile run a single job
ARGS["COMPILE"] = int(COMPILE)
if COMPILE:
    RUNS = None

# Process all subvolumes sequentially in single job
if PROCESS_ALL_SUBVOLUMES:
    RUNS = None
    ARGS["SUBVOLUME"] = "ALL"

# Set simulation and output directory
simulations = {"MR":"MillenniumWMAP1","MR7":"MillenniumWMAP7","MRII":"MillenniumIIWMAP1",\
              "MRscPlanck1":"MillenniumPlanckAnguloWhite","MRscWMAP7":"MillenniumWMAP7AnguloWhite",\
              "MRIIscPlanck1":"MillenniumIIPlanckAnguloWhite","MRIIscWMAP7":"MillenniumIIWMAP7AnguloWhite"}
if SIMULATION is None:
    raise RuntimeError("No simulation specified! Options = "+",".join(simulations.keys()))
if SIMULATION not in simulations.keys():
    raise RuntimeError("Simulation "+SIMULATION+" not valid! Options = "+",".join(simulations.keys()))
ARGS["OUTDIR"] = SCRATCH
ARGS["SIMULATION"] = SIMULATION

# Set remaing PBS options
SUBMIT.addQueue(QUEUE)
SUBMIT.addJobName(JOBNAME)
SUBMIT.specifyJobArray(RUNS)
ARGS["JOB_MANAGER"] = SUBMIT.manager
SUBMIT.passScriptArguments(ARGS)
SUBMIT.setScript(os.path.basename(__file__).replace("qsub","run"))

# Submit job
SUBMIT.printJobString()
if SUBMIT_JOB_TO_QUEUE:
    print("Submitting Galacticus job...")
    SUBMIT.submitJob()








