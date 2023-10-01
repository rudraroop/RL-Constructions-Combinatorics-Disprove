from collections import namedtuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from OptimalCuts import optimalCuts

Rectangle = namedtuple('Rectangle', field_names= ['bottomLeft', 'topRight'])

def plot_rectangles(rects, score, idx, min, max, myrand = -1):

    print(rects)
    
    fig, ax = plt.subplots()
    ax.plot([min,max], [min,max], color='white', linewidth=0)

    for rect in rects:
        ax.add_patch(patches.Rectangle(rect.bottomLeft, rect.topRight[0] - rect.bottomLeft[0], rect.topRight[1] - rect.bottomLeft[1], edgecolor = 'black', fill=False, lw=0.3))
    
    if (score > -100):
        plt.annotate('Killed ' + str(score) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    if (myrand == -1):
        fig.savefig('plots/rects_'+str(idx)+'.png')
    
    else:
        fig.savefig(f'plots/run_{str(myrand)}/rects_{str(idx)}.png')

    plt.close(fig)

def plot_rectangles_with_cuts(rects, min, max, idx):

    _, cuts, kills = optimalCuts(rects, Rectangle(bottomLeft=(min,min), topRight=(max,max)))
    print(rects)
    print("Cuts : ")
    print(cuts)
    
    fig, ax = plt.subplots()
    ax.plot([min,max], [min,max], color='white', linewidth=0)

    # Draw Rectangles
    for rect in rects:
        ax.add_patch(patches.Rectangle(rect.bottomLeft, rect.topRight[0] - rect.bottomLeft[0], rect.topRight[1] - rect.bottomLeft[1], edgecolor = 'black', fill=False, lw=0.3))
    
    # Draw Cuts
    numCuts = len(cuts)//2

    for i in range(numCuts):
        
        box = cuts[2*i]
        line = cuts[2*i + 1]

        if (line.axis == 'x'):
            height = box.topRight[1] - box.bottomLeft[1]
            anchor = (line.point, box.bottomLeft[1])
            ax.add_patch(patches.Rectangle(anchor, 0, height, edgecolor = 'red', fill=True, lw=0.3))

        else:
            width = box.topRight[0] - box.bottomLeft[0]
            anchor = (box.bottomLeft[0], line.point)
            ax.add_patch(patches.Rectangle(anchor, width, 0, edgecolor = 'red', fill=True, lw=0.3))

    ax.add_patch(patches.Rectangle((0,0), max-min, max-min, edgecolor = 'red', fill=False, lw=0.3))

    plt.annotate('Killed ' + str(kills) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    fig.savefig('plots/rects_cuts/N'+str(len(rects))+"_"+str(idx)+'.png')

    plt.close(fig)

rects = [Rectangle(bottomLeft=(0, 0), topRight=(13, 80)), Rectangle(bottomLeft=(49, 0), topRight=(104, 25)), Rectangle(bottomLeft=(95, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 14), topRight=(157, 34)), Rectangle(bottomLeft=(180, 25), topRight=(199, 98)), Rectangle(bottomLeft=(50, 45), topRight=(52, 85)), Rectangle(bottomLeft=(52, 40), topRight=(90, 90)), Rectangle(bottomLeft=(90, 45), topRight=(164, 70)), Rectangle(bottomLeft=(80, 113), topRight=(95, 179)), Rectangle(bottomLeft=(157, 71), topRight=(180, 96)), Rectangle(bottomLeft=(16, 115), topRight=(32, 135)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 78), topRight=(120, 136)), Rectangle(bottomLeft=(120, 85), topRight=(140, 184)), Rectangle(bottomLeft=(140, 99), topRight=(190, 173)), Rectangle(bottomLeft=(12, 135), topRight=(75, 160)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(108, 199)), Rectangle(bottomLeft=(21, 19), topRight=(45, 100))]
plot_rectangles_with_cuts(rects, 0, 200, 0)