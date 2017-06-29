#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 12:48:16 2017

@author: jingang
"""

import pandas as pd
import numpy as np
import os

os.chdir('/Users/jingang/Desktop/ASADataFest2017 Data')
data=pd.read_table('data.txt',nrows=1000)
dest=pd.read_table('dest.txt',nrows=1000)
