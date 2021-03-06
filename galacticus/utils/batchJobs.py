#! /usr/bin/env python

import sys,os,fnmatch,re
import numpy as np
import getpass,subprocess,glob


def getBatchVariable(variable,verbose=False,manager=None):
    if variable not in os.environ.keys():
        if manager is None:
            print("WARNING! Environment variable '"+variable+"' not found!")
        else:
            print("WARNING! "+manager+" environment variable '"+variable+"' not found!")
        value = None
    else:         
        value = os.environ[variable]
        if verbose:
            print(variable+" = "+str(value))                
    return value    
    

class JOBCLASS(object):    
    def __init__(self,manager,verbose=False):
        self.manager = manager
        self.verbose = verbose
        return

    def getEnvironmentVariable(self,variable):
        if variable not in os.environ.keys():
            if self.verbose:
                if self.manager is None:
                    print("WARNING! Environment variable '"+variable+"' not found!")
                else:
                    print("WARNING! "+self.manager+" environment variable '"+variable+"' not found!")
            value = None
        else:         
            value = os.environ[variable]
            if self.verbose:
                print(variable+" = "+str(value))                
        return value    



class NULLjob(JOBCLASS):    
    def __init__(self,verbose=False):
        super(NULLjob, self).__init__(None,verbose=verbose)
        self.interactive = True
        self.jobID = None
        self.jobArray = False
        self.jobArrayID = None
        self.jobArrayIndex = None
        self.taskID = None
        self.minJobArrayID = None
        self.maxJobArrayID = None
        self.minTaskID = None
        self.maxTaskID = None
        self.account = None
        self.userID = os.environ["USER"]
        self.machine = os.environ['HOST']
        self.queue = None
        self.jobStatus = None
        self.walltime = None
        self.nodefile = None
        self.nodes = 1
        if "OMP_NUM_THREADS" in os.environ.keys():
            self.cpus = int(os.environ["OMP_NUM_THREADS"])
        else:
            self.cpus = 1
        self.submitDir = subprocess.check_output(["pwd"]).replace("\n","")
        self.workDir = subprocess.check_output(["pwd"]).replace("\n","")
        return

    
########################################################################################################
#   SLURM classes/functions
########################################################################################################

class SLURMjob(object):
    
    def __init__(self,verbose=False):
        self.manager = "SLURM"                
        # Get job name and ID and identify whether job array
        self.jobName = getBatchVariable("SLURM_JOB_NAME",verbose=verbose,manager=self.manager)
        if self.jobName is None or self.jobName == "sh":
            self.interactive = True
        else:
            self.interactive = False
        try:
            jobID = os.environ["SLURM_ARRAY_JOB_ID"]
        except KeyError:
            self.jobID = getBatchVariable("SLURM_JOB_ID",verbose=verbose,manager=self.manager)
            self.jobArray = False
        else:
            jobID = getBatchVariable("SLURM_ARRAY_JOB_ID",verbose=verbose,manager=self.manager)
            self.jobArray = True
        self.account = getBatchVariable("SLURM_JOB_ACCOUNT",verbose=verbose,manager=self.manager)
        # Get user ID
        self.userID = getBatchVariable("SLURM_JOB_USER",verbose=verbose,manager=self.manager)
        # Get machine and queue
        self.machine = getBatchVariable("SLURM_CLUSTER_NAME",verbose=verbose,manager=self.manager)
        self.queue = getBatchVariable("SLURM_JOB_PARTITION",verbose=verbose,manager=self.manager)
        # Get submission dir
        self.submitDir = getBatchVariable("SLURM_SUBMIT_DIR",verbose=verbose,manager=self.manager)
        self.workDir = self.submitDir
        # Store variables for job array
        if self.jobArray:
            self.jobArrayID = getBatchVariable("SLURM_ARRAY_TASK_ID",verbose=verbose,manager=self.manager)
            self.taskID = self.jobArrayID
            self.minJobArrayID = getBatchVariable("SLURM_ARRAY_TASK_MIN",verbose=verbose,manager=self.manager)
            self.minTaskID = self.minJobArrayID
            self.maxJobArrayID = getBatchVariable("SLURM_ARRAY_TASK_MAX",verbose=verbose,manager=self.manager)
            self.maxTaskID = self.minJobArrayID
        else:
            self.jobArrayID = None
            self.taskID = None
            self.minJobArrayID = None
            self.maxJobArrayID = None
            self.minTaskID = None
            self.maxTaskID = None
        # Get nodes information
        self.nodes = getBatchVariable("SLURM_JOB_NUM_NODES",verbose=verbose,manager=self.manager)
        self.ppn = getBatchVariable("SLURM_JOB_CPUS_PER_NODE",verbose=verbose,manager=self.manager)
        if self.nodes is not None:
            self.nodes = int(self.nodes)
        if self.ppn is not None:                        
            if not self.ppn.isdigit():
                self.ppn = self.ppn.split("(")[0]
            self.ppn = int(self.ppn)
        if self.nodes is not None and self.ppn is not None:            
            self.cpus = self.nodes*self.ppn
        else:
            self.cpus = None
        return



def submitSLURM(script,args=None,PARTITION=None,QOS=None,WALLTIME=None,JOBNAME=None,LOGDIR=None,RUNS=None,NODETYPE=None,\
                    NODES=1,TASKS=None,CPUS=None,ACCOUNT=None,WORKDIR=None,LICENSE=None,mergeOE=False,verbose=False,submit=True):
    import sys,os,getpass,fnmatch,subprocess,glob
    sjob = "sbatch "
    if PARTITION is not None:
        sjob = sjob + " -p "+PARTITION
    if QOS is not None:
        sjob = sjob + " --qos="+QOS
    if WALLTIME is not None:
        sjob = sjob + " --time " + WALLTIME
    if JOBNAME is not None:
        sjob = sjob + " -J "+JOBNAME
    if RUNS is not None:
        if type(RUNS) is list:
            RUNS = ",".join(map(str,RUNS))
        sjob = sjob + " --array="+str(RUNS)
    if LOGDIR is not None:
        subprocess.call(["mkdir","-p",LOGDIR])        
        if JOBNAME is None:
            filename = 'slurm'
        else:
            filename = JOBNAME   
        filename = filename +'-%J'
        out = LOGDIR+"/"+filename+'.out'
        out.encode()
        err = LOGDIR+"/"+filename+'.err'
        err.encode()        
        joint = LOGDIR+"/"+filename+'.out'
        joint.encode()
        if mergeOE:
            sjob = sjob + " -i "+joint
        else:
            sjob = sjob + " --output "+out + " --error "+ err
    if ACCOUNT is not None:
        sjob = sjob + "-A " + ACCOUNT
    if NODES is None:
        NODES = 1
    sjob = sjob + " -N " + str(NODES)
    if NODETYPE is not None:
        sjob = sjob + " -C "+NODETYPE
    if TASKS is not None:
        sjob = sjob + " --ntasks-per-node="+str(TASKS)
    if CPUS is not None:
        sjob = sjob + " --cpus-per-task="+str(CPUS)
    if WORKDIR is not None:
        sjob = sjob + " --workdir="+WORKDIR
    if LICENSE is not None:
        sjob = sjob + " -L "+LICENSE
    if args is not None:
        argStr = ",".join([k+"="+args[k] for k in args.keys()])
        sjob = sjob + " --export="+argStr
    sjob = sjob + " "+script
    if verbose:
        print(" Submitting SLURM job: "+sjob)
    if submit:
        os.system(sjob)
    return




########################################################################################################
#   PBS classes/functions
########################################################################################################


class PBSjob(JOBCLASS):
    
    def __init__(self,verbose=False):
        super(PBSjob, self).__init__("PBS",verbose=verbose)
        # Get job name and ID and identify whether job array
        self.jobName = self.getEnvironmentVariable("PBS_JOBNAME")
        if self.jobName is None or self.jobName == "STDIN":
            self.interactive = True
        else:
            self.interactive = False
        # Get job ID
        self.jobID = self.getEnvironmentVariable("PBS_JOBID")
        # Get job array information
        self.jobArrayID = self.getEnvironmentVariable("PBS_ARRAY_ID")
        self.jobArrayIndex = self.getEnvironmentVariable("PBS_ARRAY_INDEX")        
        if self.jobArrayIndex:
            self.jobArray = True            
        else:
            self.jobArray = False
        self.minJobArrayID = None
        self.minTaskID = None
        self.maxJobArrayID = None
        self.maxTaskID = None
        # Get user ID
        self.userID = self.getEnvironmentVariable("PBS_O_LOGNAME")
        # Get machine and queue
        self.machine = self.getEnvironmentVariable("PBS_O_HOST")
        self.queue = self.getEnvironmentVariable("PBS_QUEUE")
        # Get submission dir
        self.submitDir = self.getEnvironmentVariable("PBS_O_WORKDIR")
        self.workDir = self.getEnvironmentVariable("PBS_O_WORKDIR")
        # Query qstat and store result
        self.jobStatus = subprocess.check_output(["qstat","-f",self.jobID]).split("\n")
        # Get nodes resource information
        self.nodefile = self.getEnvironmentVariable("PBS_NODEFILE")
        self.nodes = int(fnmatch.filter(self.jobStatus,"*Resource_List.nodect =*")[0].split("=")[-1].strip())
        self.cpus = int(fnmatch.filter(self.jobStatus,"*Resource_List.ncpus =*")[0].split("=")[-1].strip())
        self.ppn = int(float(self.cpus)/float(self.nodes))
        # Get walltime
        self.walltime = fnmatch.filter(self.jobStatus,"*Resource_List.walltime =*")[0].split("=")[-1].strip()
        return


class submitPBS(JOBCLASS):
    
    def __init__(self,verbose=False,overwrite=False):
        super(submitPBS, self).__init__("PBS",verbose=verbose)
        self.cmd = "qsub -V"
        self.appendable = True
        self.overwrite = overwrite
        return

    def canAppend(self):
        if not self.appendable:
            print("PBS submission string is not appendable!")
        return self.appendable

    def replaceOption(self,old,new):
        S = re.search(old,self.cmd)
        if S:
            self.cmd = self.cmd.replace(old,new)
        return

    def addShell(self,shell):
        if shell is None: return
        S = re.search(' -S (\w+) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Shell already specified! Shell = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -S "+shell)
        else:            
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -S "+shell+" "
        return
            
    def addQueue(self,queue):
        if queue is None: return
        S = re.search(' -q (\w+) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Queue already specified! Queue = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -q "+queue)
        else:            
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -q "+queue+" "
        return

    def addJobName(self,name):
        if name is None: return
        S = re.search(' -N (\w+) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Job name already specified! Job Name = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -N "+name)                
        else:
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -N "+name+" "
        return

    def addAccount(self,account):
        if account is None: return
        S = re.search(' -A (\w+) ',self.cmd)
        if S: 
            if not self.overwrite:
                print("Account already specified! Account = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -A "+account)                
        else:
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -A "+account+" "
        return

    def addResource(self,resourceStr):
        if not self.canAppend(): return            
        self.cmd = self.cmd + " -l "+resourceStr
        return

    def addOutputPath(self,outPath):
        if outPath is None: return
        S = re.search(' -o (\S*) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Output path already specified! Output path = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -o "+outPath)                
        else:
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -o "+outPath+" "
        return

    def addErrorPath(self,errPath):
        if errPath is None: return
        S = re.search(' -e (\S*) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Error path already specified! Error path = "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -e "+errPath)                
        else:
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -e "+errPath+" "
        return

    def joinOutErr(self):
        if not self.canAppend(): return            
        S = re.search(' -j oe ',self.cmd)
        if not S:
            self.cmd = self.cmd + " -j oe "
        return

    def specifyJobArray(self,arrayString):
        if arrayString is None: return
        S = re.search(' -J (\S*) ',self.cmd)
        if S:
            if not self.overwrite:
                print("Job array options already specified! Job array options= "+S.group(1))
                return
            else:
                self.replaceOption(S.group(0)," -J "+arrayString)                
        else:
            if not self.canAppend(): return            
            self.cmd = self.cmd + " -J "+arrayString+" "
        return

    def countJobs(self):
        nJobs = 1
        S = re.search(' -J (\S*) ',self.cmd)
        if S:
            T = re.match('(\d+)-(\d+)?(:\d+)?',S.group(1))
            start = int(T.group(1))
            end = int(T.group(2))+1
            if T.group(3):
                step = int(T.group(3).replace(":",""))               
            else:
                step = 1
            nJobs = len(np.arange(start,end,step))
        return nJobs

    def passScriptArguments(self,args):
        if args is None: return
        S = re.search(' -v (\S*) ',self.cmd)        
        if S:
            existing = {}
            for obj in S.group(1).split(","):
                existing[obj.split("=")[0]] = obj.split("=")[1]
            keys = list(map(str,list(set(args.keys() + existing.keys()))))
            argString = None
            for key in keys:
                if key in args.keys() and key in existing.keys():
                    if self.overwrite:
                        thisArg = key+"="+str(args[key])
                    else:
                        thisArg = key+"="+str(existing[key])
                else:                    
                    if key in args.keys():
                        thisArg = key+"="+str(args[key])
                    if key in existing.keys():
                        thisArg = key+"="+str(existing[key])
                if argString is None:
                    argString = thisArg
                else:
                    argString = argString+","+thisArg
            self.replaceOption(S.group(0)," -v "+argString)
        else:
            if not self.canAppend(): return            
            argString = ",".join([str(key)+"="+str(args[key]) for key in args.keys()])
            self.cmd = self.cmd + " -v "+argString+" "
        return
            
    def setScript(self,script):
        if script is None: return
        if not self.canAppend(): return            
        appendable = False
        self.cmd = self.cmd + " " +script+" "
        return

    def printJobString(self):
        print(self.cmd.replace("  "," "))
        return 

    def getJobString(self):
        return self.cmd.replace("  "," ")

    def submitJob(self):
        os.system(self.cmd.replace("  "," "))
        return

    
def submitPBS_old(script,args=None,QUEUE=None,PRIORITY=None,WALLTIME=None,JOBNAME=None,LOGDIR=None,RUNS=None,SHELL="/bin/tcsh",\
                      NODES=None,PPN=None,ACCOUNT=None,mergeOE=True,verbose=False,submit=True):
    import sys,os,getpass,fnmatch,subprocess,glob
    job = "qsub -V "
    if QUEUE is not None:
        job = job + " -q "+QUEUE
    if RUNS is not None:
        if type(RUNS) is list:
            RUNS = ",".join(map(str,RUNS))
        job = job + " -J " + str(RUNS)
    job = job + " -S "+SHELL
    if JOBNAME is not None:
        job = job + " -N "+JOBNAME
    if ACCOUNT is not None:
        job = job + "-A " + ACCOUNT        
    if WALLTIME is not None:
        job = job + " -l walltime=" + str(WALLTIME)
    if LOGDIR is not None:
        subprocess.call(["mkdir","-p",LOGDIR])                
        out = LOGDIR
        err = LOGDIR
        job = job + " -o "+out + " -e "+ err
        if mergeOE:
            job = job + " -j oe "
    if PRIORITY is not None:
        job = job + " -p "+str(PRIORITY)
    if NODES is not None and PPN is not None:
        job = job + " -l nodes=" + str(NODES) + ":ppn=" + str(PPN)
    if args is not None:
        job = job + " -v "
        firstArg = True
        for k in args.keys():
            if firstArg:
                job = job + k+"="+str(args[k])
                firstArg = False
            else:
                job = job + ","+k+"="+str(args[k])
    job = job + " "+script
    if verbose:
        print(" Submitting PBS job: "+job)
    if submit:
        os.system(job)
    return











