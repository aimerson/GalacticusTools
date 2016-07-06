#! /usr/bin/env python

import os,fnmatch
import numpy as np



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
            self.ppn = int(self.ppn)
        if self.nodes is not None and self.ppn is not None:            
            self.cpus = self.nodes*self.ppn
        else:
            self.cpus = None
        return



class PBSjob(object):
    
    def __init__(self,verbose=False):
        self.manager = "PBS"

        # Get job name and ID and identify whether job array
        self.jobName = getBatchVariable("PBS_JOBNAME",verbose=verbose,manager=self.manager)
        if self.jobName is None or self.jobName == "STDIN":
            self.interactive = True
        else:
            self.interactive = False
        try:
            jobID = os.environ["PBS_ARRAYID"]
        except KeyError:
            self.jobID = getBatchVariable("PBS_JOBID",verbose=verbose,manager=self.manager)
            self.jobArray = False
        else:
            jobID = getBatchVariable("PBS_ARRAY_ID",verbose=verbose,manager=self.manager)
            self.jobArray = True
        # Get user ID
        self.userID = getBatchVariable("PBS_O_LOGNAME",verbose=verbose,manager=self.manager)
        # Get machine and queue
        self.machine = getBatchVariable("PBS_O_HOST",verbose=verbose,manager=self.manager)
        self.queue = getBatchVariable("PBS_QUEUE",verbose=verbose,manager=self.manager)
        # Get submission dir
        self.submitDir = getBatchVariable("PBS_O_WORKDIR",verbose=verbose,manager=self.manager)
        self.workDir = getBatchVariable("PBS_O_WORKDIR",verbose=verbose,manager=self.manager)
        # Store variables for job array
        if self.jobArray:
            self.jobArrayID = getBatchVariable("PBS_ARRAY_INDEX",verbose=verbose,manager=self.manager)
            self.taskID = self.jobArrayID
            self.minJobArrayID = None
            self.minTaskID = None
            self.maxJobArrayID = None
            self.maxTaskID = None
        else:
            self.jobArrayID = None
            self.taskID = None
            self.minJobArrayID = None
            self.maxJobArrayID = None
            self.minTaskID = None
            self.maxTaskID = None
        # Get nodes information
        nodefile = getBatchVariable("PBS_NODEFILE",verbose=False,manager=self.manager)
        if nodefile is not None:
            nodes = np.loadtxt(nodefile,dtype=str)
            self.nodes = len(np.unique(nodes))        
            self.ppn = len(nodes[nodes==nodes[0]])
            self.cpus = len(nodes)        
            del nodes
        else:
            self.nodes = None
            self.ppn = None
            self.cpus = None
        return



def submitSLURM(script,args=None,PARTITION=None,QOS=None,WALLTIME=None,JOBNAME=None,LOGDIR=None,RUNS=None,\
                    NODES=1,TASKS=None,CPUS=None,ACCOUNT=None,WORKDIR=None,LICENSE=None,verbose=False,submit=True):
    import sys,os,getpass,fnmatch,subprocess,glob
    sjob = "sbatch "
    if PARTITION is not None:
        sjob = sjob + " -p "+PARTITION
    if QOS is not None:
        sjob = sjob + " --qos="+QOS
    if JOBNAME is not None:
        sjob = sjob + " -J "+JOBNAME
    if LOGDIR is None:
        pwd = subprocess.check_output(["pwd"]).replace("\n","")
        LOGDIR = pwd + "/logs"
        subprocess.call(["mkdir","-p",logdir])
    sjob = sjob + " -o "+LOGDIR + " -i "
    if WALLTIME is not None:
        sjob = sjob + " -t " + WALLTIME
    if ACCOUNT is not None:
        sjob = sjob + "-A " + ACCOUNT
    if NODES is None:
        NODES = 1
    sjob = sjob + " -N " + str(NODES)
    if TASKS is not None:
        sjob = sjob + " --ntasks-per-node="+str(TASKS)
    if CPUS is not None:
        sjob = sjob + " --cpus-per-task="+str(CPUS)
    if WORKDIR is not None:
        sjob = sjob + " --workdir="+WORKDIR
    if LICENSE is not None:
        sjob = sjob + " -L "+LICENSE
    if args is not None:
        for k in args.keys():
            sjob = sjob + " --export="+k+"="+str(args[k])
    sjob = sjob + " "+script
    if verbose:
        print(" Submitting SLURM job: "+sjob)
    if submit:
        os.system(sjob)
    return











class BATCHJOB(object):
    
    def __init__(self,verbose=False):        

        # Determine manager
        keys = os.environ.keys()
        if len(fnmatch.filter(keys,"PBS*"))>0:
            self.manager = "PBS"
        elif len(fnmatch.filter(keys,"SLURM*"))>0:
            self.manager = "SLURM"
        elif len(fnmatch.filter(keys,"LSF*"))>0:
            self.manager = "LSF"
        else:
            self.manager = None
        return



