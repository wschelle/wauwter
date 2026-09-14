#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 01:47:03 2023

@author: WauWter
"""
import os
import sys
os.chdir('/project/3017081.01/required/')
from Python.python_scripts.nii_movie import nii_movie

niifile=sys.argv[1]
nii_movie(niifile)

