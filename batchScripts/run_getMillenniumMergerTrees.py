#! /usr/bin/env python
#PBS -S /bin/tcsh
"""
run_getMillenniumMergerTrees.py -- grabs Millennium Simulation merger trees from Millennium Database

USAGE: ./run_getMillenniumMergerTrees.py [-c] [-e <exe>] [-force] [-s <simulation>] [-o <outdir>] 
                                         [-v <subvolume>] [-u <username>] [-p <password>] 
                           
"""
import os,sys,subprocess,glob
import fnmatch,re
import datetime
import numpy as np
from galacticus.utils.timing import STOPWATCH
from galacticus.utils.progress import Progress
from galacticus.utils.batchJobs import SLURMjob,PBSjob,NULLjob

# Get basic environment variables
USER = os.environ['USER']
GALACTICUS_ROOT = os.environ["GALACTICUS_ROOT_V094"]

# Initialise job class to extract necessary environment variables
JOB = None
if "JOB_MANAGER" in os.environ.keys():
    if os.environ['JOB_MANAGER'].upper() == "PBS":
        JOB = PBSjob(verbose=True)
    if os.environ['JOB_MANAGER'].upper() == "SLURM":
        JOB = SLURMjob(verbose=True)
else:
    if len(fnmatch.filter(os.environ.keys(),"PBS*"))>0:
        JOB = PBSjob(verbose=True)
    elif len(fnmatch.filter(os.environ.keys(),"SLURM*"))>0:
        JOB = PBSjob(verbose=True)
    else:
        JOB = None
if JOB is None:
    JOB = NULLjob(verbose=True)

# Get interactive jobs have arguments
if JOB.interactive and len(sys.argv) == 1:
    print(__doc__)
    quit()


#################################################################
# INPUT ARGUMENTS
#################################################################

OVERWRITE = False
OUTDIR = None
SUBVOLUME = None
COMPILE = False
SIMULATION = None
USERNAME = None
PASSWORD = None
EXE = "Millennium_Merger_Tree_File_Maker.exe"
if not JOB.interactive:
    try:
        COMPILE = bool(int(os.environ["COMPILE"]))
    except KeyError:
        pass
    try:
        OUTDIR = os.environ["OUTDIR"]
    except KeyError:
        pass
    try:
        EXE = os.environ["EXECUTABLE"]
    except KeyError:
        pass
    try:
        USERNAME = os.environ["USERNAME"]
    except KeyError:
        pass
    try:
        PASSWORD = os.environ["PASSWORD"]
    except KeyError:
        pass
    try:
        SUBVOLUME = int(os.environ["SUBVOLUME"])
    except KeyError:
        pass
    try:
        OVERWRITE = bool(int(os.environ["OVERWRITE"]))
    except KeyError:
        pass
else:
    iarg = 0
    while iarg < len(sys.argv):
        if fnmatch.fnmatch(sys.argv[iarg],"-c*"):
            COMPILE = True
        if fnmatch.fnmatch(sys.argv[iarg],"-force"):
            OVERWRITE = True
        if fnmatch.fnmatch(sys.argv[iarg],"-s*"):
            iarg += 1
            SIMULATION = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-v*"):
            iarg += 1
            SUBVOLUME = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-u*"):
            iarg += 1
            USERNAME = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-p*"):
            iarg += 1
            PASSWORD = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-o*"):
            iarg += 1
            OUTDIR = sys.argv[iarg]
        if fnmatch.fnmatch(sys.argv[iarg],"-e*"):
            iarg += 1
            EXE = sys.argv[iarg]
        iarg += 1

#################################################################

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

# Check subvolume specified
if SUBVOLUME is None:
    if JOB.jobArray:
        SUBVOLUMES = [str(JOB.jobArrayIndex-1)]
    else:
        raise RuntimeError("No subvolume(s) specified! Either specify number 1-512 or 'ALL'")
else:
    if fnmatch.fnmatch(SUBVOLUME.upper(),"ALL"):
        SUBVOLUMES = list(map(str,range(0,512)))
        if JOB.jobArray:
            raise RuntimeError("Cannot sequentially process all subvolumes in job array!")
    else:
        S = re.match("(\d+)",SUBVOLUME)
        if S:
            SUBVOLUME = int(S.group(1))
            if SUBVOLUME < 0 or SUBVOLUME > 511:
                raise RuntimeError("Subvolume must be in range 0-511.")
            SUBVOLUMES = [SUBVOLUME]
        else:
            raise RuntimeError("Subvolume "+SUBVOLUME+" not recognised!")
        
# Make sure output directory is provided
if OUTDIR is None:    
    raise RuntimeError("No output directory specified! Specify path to store simulation trees.")
# Set simulation
simulations = {"MR":"MillenniumWMAP1","MR7":"MillenniumWMAP7","MRII":"MillenniumIIWMAP1",\
              "MRscPlanck1":"MillenniumPlanckAnguloWhite","MRscWMAP7":"MillenniumWMAP7AnguloWhite",\
              "MRIIscPlanck1":"MillenniumIIPlanckAnguloWhite","MRIIscWMAP7":"MillenniumIIWMAP7AnguloWhite"}
if SIMULATION is None:
    raise RuntimeError("No simulation specified! Options = "+",".join(simulations.keys()))
if SIMULATION not in simulations.keys():
    raise RuntimeError("Simulation "+SIMULATION+" not valid! Options = "+",".join(simulations.keys()))
TABLE = "MPAHaloTrees.."+SIMULATION
OUTDIR = OUTDIR + "/MergerTrees/"+simulations[SIMULATION]+"/"    
subprocess.call(["mkdir","-p",OUTDIR])
print("SIMULATION TABLE = "+TABLE)
print("OUTPUT DIRECTORY = "+OUTDIR)

# Compile Millennium Merger Tree File Maker?
if not os.path.exists(OUTDIR+"/"+EXE):
    if os.path.exists(GALACTICUS_ROOT+"/Millennium_Merger_Tree_File_Maker.exe"):
        COMPILE = COMPILE or False
    else:
        COMPILE = COMPILE or True
if COMPILE:
    if "OMP_NUM_THREADS" in os.environ.keys():
        NCOM = os.environ["OMP_NUM_THREADS"]
    else:
        NCOM = str(JOB.ppn)
    print("-"*20)
    print("WARNING! Compiling Millennium_Merger_Tree_File_Maker!")
    S = STOPWATCH()
    print("GALACTICUS SOURCE DIRECTORY = "+GALACTICUS_ROOT)
    sys.stdout.flush()
    os.chdir(GALACTICUS_ROOT)
    sys.stdout.flush()
    os.system("make clean")
    os.system("make clean")
    os.system("make -k -j "+str(NCOM)+" Millennium_Merger_Tree_File_Maker.exe")
    if not os.path.exists("Millennium_Merger_Tree_File_Maker.exe"):
        print("*** ERROR! No Millennium Merger Tree File Maker executable found!")
        sys.exit(1)
    else:
        print("COMPILATION COMPLETED SUCCESSFULLY!")
        S.stop()
    print("-"*20)
    sys.stdout.flush()
os.system("cp "+GALACTICUS_ROOT+"/Millennium_Merger_Tree_File_Maker.exe "+OUTDIR+"/"+EXE)    

# Grab trees for specified subvolumes and convert to Galacticus format
print("Building merger trees...")
os.chdir(OUTDIR)
URL = "http://gavo.mpa-garching.mpg.de/MyMillennium?action=doQuery&SQL="
CMDBASE = "wget --http-user="+USERNAME+" --http-passwd="+PASSWORD
PROG = Progress(len(SUBVOLUMES))
for ivol in SUBVOLUMES:
    outfile = "subvolume"+str(ivol)+".csv"
    if not os.path.exists(outfile) or OVERWRITE:
        PROPS = ["node.treeID","node.haloId","node.descendantId","node.firstHaloInFOFgroupId",\
                 "node.snapNum","node.redshift","node.m_tophat","node.m_mean200","node.m_crit200",\
                 "node.np","node.x","node.y","node.z","node.velX","node.velY","node.velZ",\
                 "node.spinX","node.spinY","node.spinZ","node.halfmassRadius","node.mostBoundID",\
                 "node.vMax","node.vDisp"]
        SQL = URL+"select "+", ".join(PROPS)+" from "+TABLE+" node "
        SQL = SQL + "where node.treeRootId between "+str(ivol)+"000000000000 and "+str(ivol)+"999999999999 order by node.treeId"
        CMD = CMDBASE+" \""+SQL+"\" -O "+outfile
        os.system(CMD)
    treefile = "subvolume"+str(ivol)+".hdf5"
    if not os.path.exists(treefile) or OVERWRITE:
        CMD = "./"+EXE+" "+outfile+" none "+treefile+" galacticus 1"
        print CMD
        os.system(CMD)
    PROG.increment()
    PROG.print_status_line("subvolume = "+str(ivol))

        
    
