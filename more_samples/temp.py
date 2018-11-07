# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from pandas import ExcelFile


d = os.listdir('segments')

path_measures = 'man_jacket_hand_measures.xls'
xl = ExcelFile(path_measures)
sheet = xl.parse(xl.sheet_names[0])
""" be careful, parse() just reads literals, does not execute formulas """
xl.close()

it = sheet.iterrows()
ides_xls = []
for row in it:
    ides_xls.append(row[1]['ide'])

ides_files = [fname.split('_') for fname in d]

for fname in d:
    if not(fname.split('_')[0] in ides_xls):
        os.rename(os.path.join('segments',fname),
                  os.path.join('segments','no_'+fname))