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

#rects = [Rectangle(bottomLeft=(0, 55), topRight=(35, 85)), Rectangle(bottomLeft=(35, 13), topRight=(120, 70)), Rectangle(bottomLeft=(120, 14), topRight=(130, 85)), Rectangle(bottomLeft=(130, 0), topRight=(159, 65)), Rectangle(bottomLeft=(159, 0), topRight=(188, 65)), Rectangle(bottomLeft=(26, 85), topRight=(36, 98)), Rectangle(bottomLeft=(60, 99), topRight=(105, 107)), Rectangle(bottomLeft=(98, 79), topRight=(107, 83)), Rectangle(bottomLeft=(130, 65), topRight=(140, 108)), Rectangle(bottomLeft=(140, 75), topRight=(199, 115)), Rectangle(bottomLeft=(0, 119), topRight=(76, 160)), Rectangle(bottomLeft=(36, 73), topRight=(60, 88)), Rectangle(bottomLeft=(76, 155), topRight=(97, 179)), Rectangle(bottomLeft=(105, 103), topRight=(129, 162)), Rectangle(bottomLeft=(181, 128), topRight=(199, 159)), Rectangle(bottomLeft=(34, 160), topRight=(60, 199)), Rectangle(bottomLeft=(60, 179), topRight=(69, 199)), Rectangle(bottomLeft=(70, 179), topRight=(107, 199)), Rectangle(bottomLeft=(107, 162), topRight=(192, 199))]
#plot_rectangles_with_cuts(rects, 0, 200, 0)