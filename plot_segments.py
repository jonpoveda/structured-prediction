# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:19:37 2017

@author: joans
"""
import matplotlib.pyplot as plt


def plot_segments(segments, caption='', labels_segments=None):
    colors = 'rgbcmyk'*2
    fig = plt.figure()
    num_segments = len(segments)
    for s,n in zip(segments, range(num_segments)):
        if labels_segments is None:
            color = 'blue'
        else:
            color = colors[labels_segments[n]]
        
        plt.plot([s.y0, s.y1], [s.x0, s.x1], 'o-', color=color,linewidth=2)                    
        plt.text( (s.y0+s.y1)/2, (s.x0+s.x1)/2, str(n), color='k')
    
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.title(caption)
    plt.show(block=False)

    return fig
