from collections import namedtuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from OptimalCuts import optimalCuts, Rectangle

def get_pinwheels(rects, min, max):

    num_rects = len(rects)
    pinwheels = []

    print("Kills: ")
    for i in range(0, num_rects-3):
        for j in range(i+1, num_rects-2):
            for k in range(j+1, num_rects-1):
                for l in range(k+1, num_rects):
                    _, cuts, kills = optimalCuts([rects[i], rects[j], rects[k], rects[l]], Rectangle(bottomLeft=(min,min), topRight=(max,max)))
                    if (kills > 0):
                        print(kills)
                        pinwheels.append([i, j, k, l])
    
    return pinwheels

def plot_rects_w_color_pinwheel(rects, pinwheel, kills, min, max, construction_idx, pinwheel_idx):

    print(rects)
    
    fig, ax = plt.subplots()
    ax.plot([min,max], [min,max], color='white', linewidth=0)

    for rect in rects:
        ax.add_patch(patches.Rectangle(rect.bottomLeft, rect.topRight[0] - rect.bottomLeft[0], rect.topRight[1] - rect.bottomLeft[1], edgecolor = 'black', fill=False, lw=0.2))

    for index in pinwheel:
        ax.add_patch(patches.Rectangle(rects[index].bottomLeft, rects[index].topRight[0] - rects[index].bottomLeft[0], rects[index].topRight[1] - rects[index].bottomLeft[1], edgecolor = 'green', fill=False, lw=1.3))
    
    if (kills >= 0):
        plt.annotate('Killed ' + str(kills) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, f'pinwheel_plots/N_{str(len(rects))}/construction_{str(construction_idx)}/')
    
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    fig.savefig(results_dir + f'pinwheel_{str(pinwheel_idx)}.png')
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

def plot_all_pinwheels(rects, min, max, kills, construction_idx):
    
    pinwheels = get_pinwheels(rects, min, max)

    for i in range(len(pinwheels)):
        plot_rects_w_color_pinwheel(rects, pinwheels[i], kills, min, max, construction_idx, i)

def eliminatePinwheels(rects, min, max):
    # will return list of rectangle indexes - containing minimum number of rectangles we need to eliminate to get rid of all pinwheels
    pinwheels = get_pinwheels(rects, min, max)
    print(f"Pinwheels are: {str(pinwheels)}")

    for i1 in range(4):
        for i2 in range(4):
            rects_new = list(rects)
            del rects_new[pinwheels[1][i2]]
            del rects_new[pinwheels[0][i1]]
            plot_rectangles_with_cuts(rects_new, 0, 200, (4*i1) + i2)

all_18_rect_sets = [[Rectangle(bottomLeft=(0, 45), topRight=(16, 64)), Rectangle(bottomLeft=(50, 0), topRight=(105, 25)), Rectangle(bottomLeft=(90, 25), topRight=(105, 45)), Rectangle(bottomLeft=(114, 0), topRight=(180, 34)), Rectangle(bottomLeft=(180, 26), topRight=(199, 46)), Rectangle(bottomLeft=(16, 64), topRight=(52, 115)), Rectangle(bottomLeft=(52, 63), topRight=(90, 113)), Rectangle(bottomLeft=(90, 45), topRight=(153, 78)), Rectangle(bottomLeft=(80, 113), topRight=(95, 168)), Rectangle(bottomLeft=(153, 34), topRight=(180, 82)), Rectangle(bottomLeft=(27, 115), topRight=(43, 135)), Rectangle(bottomLeft=(43, 115), topRight=(80, 131)), Rectangle(bottomLeft=(95, 114), topRight=(120, 178)), Rectangle(bottomLeft=(120, 78), topRight=(149, 156)), Rectangle(bottomLeft=(149, 82), topRight=(199, 127)), Rectangle(bottomLeft=(17, 135), topRight=(80, 160)), Rectangle(bottomLeft=(49, 160), topRight=(70, 189)), Rectangle(bottomLeft=(75, 179), topRight=(136, 199))], [Rectangle(bottomLeft=(29, 0), topRight=(42, 85)), Rectangle(bottomLeft=(42, 0), topRight=(105, 23)), Rectangle(bottomLeft=(95, 23), topRight=(105, 46)), Rectangle(bottomLeft=(105, 17), topRight=(175, 45)), Rectangle(bottomLeft=(180, 26), topRight=(198, 99)), Rectangle(bottomLeft=(50, 75), topRight=(52, 115)), Rectangle(bottomLeft=(52, 63), topRight=(90, 113)), Rectangle(bottomLeft=(90, 46), topRight=(134, 85)), Rectangle(bottomLeft=(80, 113), topRight=(95, 133)), Rectangle(bottomLeft=(146, 72), topRight=(180, 99)), Rectangle(bottomLeft=(0, 86), topRight=(8, 112)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 85), topRight=(120, 149)), Rectangle(bottomLeft=(120, 85), topRight=(132, 161)), Rectangle(bottomLeft=(132, 99), topRight=(199, 120)), Rectangle(bottomLeft=(0, 135), topRight=(80, 148)), Rectangle(bottomLeft=(60, 148), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(120, 186))]
, [Rectangle(bottomLeft=(0, 0), topRight=(17, 89)), Rectangle(bottomLeft=(55, 0), topRight=(123, 25)), Rectangle(bottomLeft=(95, 25), topRight=(117, 45)), Rectangle(bottomLeft=(117, 25), topRight=(180, 45)), Rectangle(bottomLeft=(180, 0), topRight=(199, 48)), Rectangle(bottomLeft=(50, 0), topRight=(55, 5)), Rectangle(bottomLeft=(55, 62), topRight=(87, 82)), Rectangle(bottomLeft=(93, 45), topRight=(136, 70)), Rectangle(bottomLeft=(80, 82), topRight=(95, 132)), Rectangle(bottomLeft=(140, 45), topRight=(163, 95)), Rectangle(bottomLeft=(16, 89), topRight=(53, 115)), Rectangle(bottomLeft=(32, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 121), topRight=(102, 171)), Rectangle(bottomLeft=(120, 70), topRight=(140, 145)), Rectangle(bottomLeft=(140, 96), topRight=(199, 187)), Rectangle(bottomLeft=(17, 135), topRight=(80, 160)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(111, 187))]
, [Rectangle(bottomLeft=(0, 0), topRight=(49, 44)), Rectangle(bottomLeft=(49, 0), topRight=(99, 25)), Rectangle(bottomLeft=(92, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 0), topRight=(168, 41)), Rectangle(bottomLeft=(180, 0), topRight=(199, 83)), Rectangle(bottomLeft=(31, 44), topRight=(52, 71)), Rectangle(bottomLeft=(52, 40), topRight=(90, 50)), Rectangle(bottomLeft=(90, 45), topRight=(155, 70)), Rectangle(bottomLeft=(80, 81), topRight=(95, 131)), Rectangle(bottomLeft=(157, 71), topRight=(180, 98)), Rectangle(bottomLeft=(0, 115), topRight=(43, 131)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(99, 105), topRight=(144, 179)), Rectangle(bottomLeft=(124, 70), topRight=(144, 99)), Rectangle(bottomLeft=(144, 98), topRight=(199, 148)), Rectangle(bottomLeft=(12, 135), topRight=(62, 160)), Rectangle(bottomLeft=(31, 160), topRight=(70, 199)), Rectangle(bottomLeft=(75, 179), topRight=(125, 195))]
, [Rectangle(bottomLeft=(0, 0), topRight=(5, 68)), Rectangle(bottomLeft=(49, 0), topRight=(105, 25)), Rectangle(bottomLeft=(95, 25), topRight=(103, 45)), Rectangle(bottomLeft=(105, 14), topRight=(155, 39)), Rectangle(bottomLeft=(180, 25), topRight=(199, 75)), Rectangle(bottomLeft=(50, 56), topRight=(52, 106)), Rectangle(bottomLeft=(52, 63), topRight=(90, 113)), Rectangle(bottomLeft=(90, 45), topRight=(140, 78)), Rectangle(bottomLeft=(80, 113), topRight=(95, 123)), Rectangle(bottomLeft=(140, 39), topRight=(180, 94)), Rectangle(bottomLeft=(13, 73), topRight=(50, 114)), Rectangle(bottomLeft=(32, 114), topRight=(80, 135)), Rectangle(bottomLeft=(95, 121), topRight=(120, 165)), Rectangle(bottomLeft=(120, 78), topRight=(140, 161)), Rectangle(bottomLeft=(140, 96), topRight=(177, 182)), Rectangle(bottomLeft=(0, 135), topRight=(80, 160)), Rectangle(bottomLeft=(30, 160), topRight=(70, 191)), Rectangle(bottomLeft=(70, 179), topRight=(140, 199))]
, [Rectangle(bottomLeft=(0, 0), topRight=(13, 81)), Rectangle(bottomLeft=(49, 0), topRight=(99, 25)), Rectangle(bottomLeft=(95, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 14), topRight=(180, 45)), Rectangle(bottomLeft=(180, 25), topRight=(199, 74)), Rectangle(bottomLeft=(13, 45), topRight=(26, 95)), Rectangle(bottomLeft=(26, 25), topRight=(76, 46)), Rectangle(bottomLeft=(90, 45), topRight=(172, 70)), Rectangle(bottomLeft=(80, 113), topRight=(95, 179)), Rectangle(bottomLeft=(148, 73), topRight=(149, 98)), Rectangle(bottomLeft=(16, 95), topRight=(32, 135)), Rectangle(bottomLeft=(32, 90), topRight=(95, 113)), Rectangle(bottomLeft=(95, 82), topRight=(118, 179)), Rectangle(bottomLeft=(120, 70), topRight=(137, 95)), Rectangle(bottomLeft=(140, 98), topRight=(185, 103)), Rectangle(bottomLeft=(0, 135), topRight=(75, 160)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(145, 193))]
, [Rectangle(bottomLeft=(0, 0), topRight=(45, 78)), Rectangle(bottomLeft=(49, 0), topRight=(99, 25)), Rectangle(bottomLeft=(90, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 20), topRight=(161, 27)), Rectangle(bottomLeft=(180, 25), topRight=(199, 76)), Rectangle(bottomLeft=(45, 45), topRight=(52, 90)), Rectangle(bottomLeft=(52, 40), topRight=(90, 65)), Rectangle(bottomLeft=(90, 45), topRight=(180, 70)), Rectangle(bottomLeft=(80, 113), topRight=(95, 168)), Rectangle(bottomLeft=(157, 70), topRight=(180, 82)), Rectangle(bottomLeft=(0, 80), topRight=(31, 130)), Rectangle(bottomLeft=(32, 90), topRight=(82, 113)), Rectangle(bottomLeft=(95, 70), topRight=(120, 120)), Rectangle(bottomLeft=(120, 70), topRight=(140, 120)), Rectangle(bottomLeft=(140, 98), topRight=(197, 148)), Rectangle(bottomLeft=(17, 135), topRight=(67, 160)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(95, 199))]
, [Rectangle(bottomLeft=(15, 0), topRight=(24, 56)), Rectangle(bottomLeft=(49, 0), topRight=(105, 15)), Rectangle(bottomLeft=(90, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 14), topRight=(180, 45)), Rectangle(bottomLeft=(180, 21), topRight=(199, 71)), Rectangle(bottomLeft=(50, 45), topRight=(52, 103)), Rectangle(bottomLeft=(52, 40), topRight=(90, 90)), Rectangle(bottomLeft=(90, 45), topRight=(163, 55)), Rectangle(bottomLeft=(80, 90), topRight=(95, 152)), Rectangle(bottomLeft=(157, 70), topRight=(180, 99)), Rectangle(bottomLeft=(16, 76), topRight=(50, 91)), Rectangle(bottomLeft=(43, 103), topRight=(74, 135)), Rectangle(bottomLeft=(95, 78), topRight=(120, 175)), Rectangle(bottomLeft=(120, 85), topRight=(140, 135)), Rectangle(bottomLeft=(140, 99), topRight=(190, 131)), Rectangle(bottomLeft=(17, 135), topRight=(67, 160)), Rectangle(bottomLeft=(20, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 175), topRight=(120, 178))]
, [Rectangle(bottomLeft=(0, 26), topRight=(50, 76)), Rectangle(bottomLeft=(50, 0), topRight=(100, 23)), Rectangle(bottomLeft=(95, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 0), topRight=(152, 45)), Rectangle(bottomLeft=(180, 25), topRight=(181, 96)), Rectangle(bottomLeft=(50, 51), topRight=(52, 140)), Rectangle(bottomLeft=(52, 40), topRight=(90, 73)), Rectangle(bottomLeft=(90, 45), topRight=(179, 70)), Rectangle(bottomLeft=(80, 113), topRight=(91, 169)), Rectangle(bottomLeft=(157, 70), topRight=(176, 79)), Rectangle(bottomLeft=(3, 77), topRight=(26, 129)), Rectangle(bottomLeft=(43, 140), topRight=(80, 142)), Rectangle(bottomLeft=(91, 84), topRight=(120, 171)), Rectangle(bottomLeft=(120, 70), topRight=(139, 136)), Rectangle(bottomLeft=(149, 99), topRight=(181, 134)), Rectangle(bottomLeft=(0, 135), topRight=(25, 149)), Rectangle(bottomLeft=(30, 160), topRight=(54, 199)), Rectangle(bottomLeft=(54, 171), topRight=(104, 199))]
, [Rectangle(bottomLeft=(0, 0), topRight=(13, 45)), Rectangle(bottomLeft=(17, 0), topRight=(79, 29)), Rectangle(bottomLeft=(95, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 14), topRight=(199, 34)), Rectangle(bottomLeft=(180, 34), topRight=(199, 87)), Rectangle(bottomLeft=(13, 29), topRight=(32, 115)), Rectangle(bottomLeft=(52, 40), topRight=(95, 90)), Rectangle(bottomLeft=(99, 45), topRight=(149, 78)), Rectangle(bottomLeft=(80, 113), topRight=(120, 179)), Rectangle(bottomLeft=(149, 73), topRight=(180, 98)), Rectangle(bottomLeft=(16, 115), topRight=(43, 130)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 78), topRight=(120, 79)), Rectangle(bottomLeft=(120, 96), topRight=(134, 146)), Rectangle(bottomLeft=(146, 98), topRight=(154, 180)), Rectangle(bottomLeft=(0, 135), topRight=(50, 173)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(146, 199))]
, [Rectangle(bottomLeft=(0, 0), topRight=(29, 50)), Rectangle(bottomLeft=(29, 15), topRight=(92, 40)), Rectangle(bottomLeft=(95, 25), topRight=(105, 78)), Rectangle(bottomLeft=(105, 14), topRight=(155, 29)), Rectangle(bottomLeft=(180, 0), topRight=(187, 59)), Rectangle(bottomLeft=(49, 75), topRight=(57, 84)), Rectangle(bottomLeft=(57, 40), topRight=(58, 78)), Rectangle(bottomLeft=(106, 45), topRight=(156, 74)), Rectangle(bottomLeft=(80, 113), topRight=(95, 179)), Rectangle(bottomLeft=(157, 71), topRight=(180, 91)), Rectangle(bottomLeft=(16, 82), topRight=(49, 115)), Rectangle(bottomLeft=(38, 115), topRight=(53, 130)), Rectangle(bottomLeft=(95, 78), topRight=(115, 128)), Rectangle(bottomLeft=(120, 74), topRight=(140, 124)), Rectangle(bottomLeft=(140, 98), topRight=(165, 114)), Rectangle(bottomLeft=(12, 135), topRight=(80, 160)), Rectangle(bottomLeft=(47, 160), topRight=(61, 199)), Rectangle(bottomLeft=(70, 179), topRight=(120, 199))]
, [Rectangle(bottomLeft=(27, 0), topRight=(49, 59)), Rectangle(bottomLeft=(49, 0), topRight=(105, 25)), Rectangle(bottomLeft=(52, 25), topRight=(105, 46)), Rectangle(bottomLeft=(105, 6), topRight=(143, 31)), Rectangle(bottomLeft=(180, 14), topRight=(199, 64)), Rectangle(bottomLeft=(50, 25), topRight=(52, 63)), Rectangle(bottomLeft=(52, 62), topRight=(66, 107)), Rectangle(bottomLeft=(90, 46), topRight=(180, 47)), Rectangle(bottomLeft=(80, 113), topRight=(95, 163)), Rectangle(bottomLeft=(130, 47), topRight=(180, 97)), Rectangle(bottomLeft=(27, 115), topRight=(43, 127)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 78), topRight=(120, 118)), Rectangle(bottomLeft=(120, 85), topRight=(130, 105)), Rectangle(bottomLeft=(140, 99), topRight=(190, 137)), Rectangle(bottomLeft=(12, 135), topRight=(66, 160)), Rectangle(bottomLeft=(28, 160), topRight=(66, 199)), Rectangle(bottomLeft=(66, 179), topRight=(116, 199))]
, [Rectangle(bottomLeft=(0, 0), topRight=(49, 39)), Rectangle(bottomLeft=(49, 0), topRight=(105, 25)), Rectangle(bottomLeft=(95, 25), topRight=(105, 78)), Rectangle(bottomLeft=(105, 14), topRight=(180, 29)), Rectangle(bottomLeft=(180, 25), topRight=(199, 75)), Rectangle(bottomLeft=(20, 39), topRight=(52, 115)), Rectangle(bottomLeft=(52, 40), topRight=(95, 90)), Rectangle(bottomLeft=(106, 42), topRight=(156, 71)), Rectangle(bottomLeft=(80, 90), topRight=(95, 140)), Rectangle(bottomLeft=(149, 71), topRight=(180, 96)), Rectangle(bottomLeft=(16, 115), topRight=(32, 124)), Rectangle(bottomLeft=(32, 115), topRight=(80, 118)), Rectangle(bottomLeft=(95, 78), topRight=(98, 139)), Rectangle(bottomLeft=(120, 85), topRight=(149, 182)), Rectangle(bottomLeft=(149, 96), topRight=(199, 121)), Rectangle(bottomLeft=(12, 135), topRight=(44, 185)), Rectangle(bottomLeft=(44, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 155), topRight=(120, 178))]
, [Rectangle(bottomLeft=(0, 0), topRight=(49, 39)), Rectangle(bottomLeft=(49, 0), topRight=(105, 25)), Rectangle(bottomLeft=(95, 25), topRight=(105, 78)), Rectangle(bottomLeft=(105, 14), topRight=(180, 29)), Rectangle(bottomLeft=(180, 25), topRight=(199, 75)), Rectangle(bottomLeft=(20, 39), topRight=(52, 115)), Rectangle(bottomLeft=(52, 40), topRight=(95, 90)), Rectangle(bottomLeft=(106, 42), topRight=(156, 71)), Rectangle(bottomLeft=(80, 90), topRight=(95, 140)), Rectangle(bottomLeft=(149, 71), topRight=(180, 96)), Rectangle(bottomLeft=(16, 115), topRight=(32, 124)), Rectangle(bottomLeft=(32, 115), topRight=(80, 118)), Rectangle(bottomLeft=(95, 78), topRight=(98, 139)), Rectangle(bottomLeft=(120, 85), topRight=(149, 182)), Rectangle(bottomLeft=(149, 96), topRight=(199, 121)), Rectangle(bottomLeft=(12, 135), topRight=(44, 185)), Rectangle(bottomLeft=(44, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 155), topRight=(120, 178))]]

#for i in range(len(all_18_rect_sets)):
#    plot_all_pinwheels(all_18_rect_sets[i], 0, 200, 4, i)

print(get_pinwheels(all_18_rect_sets[11], 0, 200))

eliminatePinwheels(all_18_rect_sets[11], 0, 200)

#rects_new = list(all_18_rect_sets[11])
#del rects_new[11]
#del rects_new[1]

#pw = get_pinwheels(rects_new, 0, 200)

#plot_all_pinwheels(rects_new, 0, 200, 4, 0)
#plot_rectangles_with_cuts(all_18_rect_sets[11], 0, 200, 1)

def get_pinwheels_5(rects, min, max):

    num_rects = len(rects)
    pinwheels = []

    print("Kills: ")
    for m in range(0, num_rects-4):
        for i in range(m+1, num_rects-3):
            for j in range(i+1, num_rects-2):
                for k in range(j+1, num_rects-1):
                    for l in range(k+1, num_rects):
                        _, cuts, kills = optimalCuts([rects[m], rects[i], rects[j], rects[k], rects[l]], Rectangle(bottomLeft=(min,min), topRight=(max,max)))
                        if (kills > 0):
                            print(kills)
                            pinwheels.append([m, i, j, k, l])
    
    return pinwheels

#pw5 = get_pinwheels_5(rects_new, 0, 200)
#print(f"New Pinwheels ====== {pw5}")
