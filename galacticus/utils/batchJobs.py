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
        if self.jobName == "sh":
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
        self.nodes = int(getBatchVariable("SLURM_JOB_NUM_NODES",verbose=verbose,manager=self.manager))
        self.ppn = int(getBatchVariable("SLURM_JOB_CPUS_PER_NODE",verbose=verbose,manager=self.manager))
        self.cpus = int(self.nodes)*int(self.ppn)
        return



class PBSjob(object):
    
    def __init__(self,verbose=False):
        self.manager = "PBS"

        # Get job name and ID and identify whether job array
        self.jobName = getBatchVariable("PBS_JOBNAME",verbose=verbose,manager=self.manager)
        if self.jobName == "STDIN":
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
        nodes = np.loadtxt(os.environ["PBS_NODEFILE"],dtype=str)
        self.nodes = len(np.unique(nodes))        
        self.ppn = len(nodes[nodes==nodes[0]])
        self.cpus = len(nodes)        
        del nodes
               
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



