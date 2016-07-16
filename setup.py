#! /usr/bin/env python

from setuptools import setup, find_packages

setup(name='galacticus',
      version='0.1',
      description='User tools for the Galacticus semi-annalytical model',
      url='http://bitbucket.org/aimerson/galacticustools',
      author='Alex Merson',
      author_email='alex.i.merson@gmail.com',
      license='MIT',
      packages=find_packages(),
      package_dir={'galacticus':'galacticus'},
      package_data={'galacticus':['data/LuminosityFunctions/*.dat']},
      zip_safe=False)

