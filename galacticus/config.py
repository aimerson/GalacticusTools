#! /usr/bin/env python

import os,subprocess

#############################
# GALACTICUS PATH
#############################
try:
    galacticusPath = os.environ["GALACTICUS_ROOT_V094"]
except KeyError:
    galacticusPath = subprocess.check_output(["pwd"]).replace("\n","")




