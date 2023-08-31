import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_rectangles(rects, score, idx, min, max):
    
    fig, ax = plt.subplots()
    ax.plot([min,max], [min,max], color='white', linewidth=0)

    for rect in rects:
        ax.add_patch(patches.Rectangle(rect.bottomLeft, rect.topRight[0] - rect.bottomLeft[0], rect.topRight[1] - rect.bottomLeft[1], edgecolor = 'black', fill=False, lw=0.3))
    
    if (score > -100):
        plt.annotate('Killed ' + str(score) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    fig.savefig('plots/rects_'+str(idx)+'.png')
    plt.close(fig)