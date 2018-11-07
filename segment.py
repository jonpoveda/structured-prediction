# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 00:36:11 2017

@author: joans
"""
import numpy as np

class Segment:  
    def __init__(self, x0, y0, x1, y1,
                 x0norm, y0norm, x1norm, y1norm):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.x0norm = x0norm
        self.y0norm = y0norm
        self.x1norm = x1norm
        self.y1norm = y1norm
        self.angle = np.arctan2((x1-x0),(y1-y0+1.0e-5))
        self.length = np.sqrt((x0-x1)**2 + (y0-y1)**2)
 