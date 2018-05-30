#! /usr/bin/env python


class DatasetClass(object):

    def __init__(self,datasetName=None,redshift=None,outputName=None):
        self.datasetName = datasetName
        self.redshift = redshift
        self.outputName = outputName
        return
