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


    plt.annotate('Killed ' + str(kills) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    fig.savefig('plots/rects_cuts/N'+str(len(rects))+"_"+str(idx)+'.png')

    plt.close(fig)

#rects = [Rectangle(bottomLeft=(25, 72), topRight=(44, 85)), Rectangle(bottomLeft=(35, 0), topRight=(104, 39)), Rectangle(bottomLeft=(105, 0), topRight=(188, 25)), Rectangle(bottomLeft=(105, 25), topRight=(165, 70)), Rectangle(bottomLeft=(165, 25), topRight=(199, 47)), Rectangle(bottomLeft=(0, 45), topRight=(24, 115)), Rectangle(bottomLeft=(44, 79), topRight=(121, 99)), Rectangle(bottomLeft=(121, 70), topRight=(131, 110)), Rectangle(bottomLeft=(131, 80), topRight=(181, 86)), Rectangle(bottomLeft=(189, 47), topRight=(199, 116)), Rectangle(bottomLeft=(0, 115), topRight=(42, 123)), Rectangle(bottomLeft=(42, 99), topRight=(67, 124)), Rectangle(bottomLeft=(67, 101), topRight=(112, 184)), Rectangle(bottomLeft=(112, 117), topRight=(127, 177))]
#plot_rectangles_with_cuts(rects, 0, 200, 0)

#[Rectangle(bottomLeft=(25, 0), topRight=(35, 9)), Rectangle(bottomLeft=(35, 11), topRight=(84, 70)), Rectangle(bottomLeft=(134, 42), topRight=(146, 81)), Rectangle(bottomLeft=(105, 2), topRight=(176, 25)), Rectangle(bottomLeft=(189, 3), topRight=(199, 77)), Rectangle(bottomLeft=(0, 49), topRight=(35, 115)), Rectangle(bottomLeft=(55, 74), topRight=(73, 113)), Rectangle(bottomLeft=(73, 93), topRight=(136, 113)), Rectangle(bottomLeft=(136, 95), topRight=(189, 115)), Rectangle(bottomLeft=(146, 25), topRight=(165, 80)), Rectangle(bottomLeft=(37, 142), topRight=(47, 199)), Rectangle(bottomLeft=(47, 113), topRight=(88, 146)), Rectangle(bottomLeft=(118, 162), topRight=(130, 187)), Rectangle(bottomLeft=(145, 165), topRight=(199, 182)), Rectangle(bottomLeft=(165, 115), topRight=(167, 165)), Rectangle(bottomLeft=(0, 179), topRight=(37, 199)), Rectangle(bottomLeft=(47, 179), topRight=(109, 182)), Rectangle(bottomLeft=(107, 124), topRight=(165, 160)), Rectangle(bottomLeft=(135, 160), topRight=(145, 177))]
