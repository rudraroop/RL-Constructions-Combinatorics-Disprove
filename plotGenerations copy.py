from collections import namedtuple
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from OptimalCuts import optimalCuts

# max number of cut through rectangles - this is a very safe upper bound as we are not going beyond n = 20
INF = 10000

# Rectangles are represented as a tuple (bottomLeft = (x0, y0), topRight = (x1, y1))
Rectangle = namedtuple('Rectangle', field_names= ['bottomLeft', 'topRight'])

# Axis Parallel Lines are represented as (point, axis) where point is a number and axis is a character
# If axis == 'x', line -> x = point || axis == 'y', line -> y = point 
AxisParallelLine = namedtuple('AxisParallelLine', field_names=['point', 'axis'])

# Interval is simply Represented as a tuple (start, end) - this will not be a named tuple

# Boundary sharing intervals are not considered to be intersecting
def intervalsIntersect(int1, int2):
    return not (int1[0] >= int2[1] or int2[0] >= int1[1])

def lineIntersectsRectangle(line, rect):
    if (line.axis == 'x'):
        return intervalsIntersect((rect.bottomLeft[0], rect.topRight[0]), (line.point, line.point))
    else:
        return intervalsIntersect((rect.bottomLeft[1], rect.topRight[1]), (line.point, line.point))

def isDisjoint(rect1, rect2):
    xInterval1, xInterval2, yInterval1, yInterval2 = (rect1.bottomLeft[0], rect1.topRight[0]), (rect2.bottomLeft[0], rect2.topRight[0]), (rect1.bottomLeft[1], rect1.topRight[1]), (rect2.bottomLeft[1], rect2.topRight[1])
    return not (intervalsIntersect(xInterval1, xInterval2) and intervalsIntersect(yInterval1, yInterval2))

def optimalCuts(rects, reg):
    
    # rects is a list of all the rectangles
    # reg is a rectangle representing the bounded region containing all the rectangles

    # returns (Pairwise Disjoint Validity, Optimal Cut Sequence, Number of Rectangles Killed)

    # check if rectangle set is pairwise disjoint
    for i in range(len(rects)):
        for j in range(i+1, len(rects)):
            if not isDisjoint(rects[i], rects[j]):
                return (False, [], 0)
            
    memo = {}  # store DP Results

    # make sorted list of x and y coordinates of all rectangle boundary points
    x = set()
    y = set()

    for rectangle in rects:
        x.add(rectangle.bottomLeft[0])
        x.add(rectangle.topRight[0])
        y.add(rectangle.bottomLeft[1])
        y.add(rectangle.topRight[1])

    x = sorted(list(x))
    y = sorted(list(y))

    seq = []

    result = findOptimalCuts(rects, x, y, reg, seq, memo)
    return (True, result[0], result[1])

def findOptimalCuts(rects, x, y, reg, seq, memo):

    if len(rects) <= 3:
        return seq, 0
    
    # check if memoized
    if reg in memo:
        return memo[reg]
    
    # Try making a guillotine cut along every rectangle boundary except vertical and horizontal borders
    numCandidates = len(x) + len(y) - 4
    cuts = [ INF for i in range(numCandidates) ]	
    sequences = []

    # Iterate through all vertical rectangle boundaries (except borders)
    for i in range( 1 , len(x) - 1 ):
        
        proposedCut = AxisParallelLine( point = x[i], axis = 'x' ) 
        rectsLeft, rectsRight = set(), set()    # boundary will split region into left and right subregions
        rectStartingAtBoundary = False
        currentKilled = 0

        #check what the guillotine cut does to each rectangle in the region
        for rect in rects: 

            if lineIntersectsRectangle( proposedCut , rect ):
                # this rectangle is killed
                currentKilled += 1
            elif rect.topRight[0] <= x[i]:
                # rectangle falls in the left subregion
                rectsLeft.add(rect)
            else:
                rectsRight.add(rect)
        
            if rect.bottomLeft[0] == x[i]: 
                # the guillotine cut has a boundary starting rectangle
                rectStartingAtBoundary = True

        xLeft = x[ : i+1]
        xRight = x[i : ] if rectStartingAtBoundary else x[i + 1 : ]

        yLeft = set()
        for rect in rectsLeft:
            yLeft.add(rect.bottomLeft[1])
            yLeft.add(rect.topRight[1])
        yLeft = sorted(list(yLeft))

        yRight = set()
        for rect in rectsRight:
            yRight.add(rect.bottomLeft[1])
            yRight.add(rect.topRight[1])
        yRight = sorted(list(yRight))

        regionLeft = Rectangle( bottomLeft = reg.bottomLeft, topRight = (x[i], reg.topRight[1]) )
        regionRight = Rectangle( bottomLeft = (x[i], reg.bottomLeft[1]), topRight = reg.topRight )

        seqLeft, killedLeft = findOptimalCuts(rectsLeft, xLeft, yLeft, regionLeft, seq, memo)
        seqRight, killedRight = findOptimalCuts(rectsRight, xRight, yRight, regionRight, seq, memo)

        cuts[i-1] = killedLeft + killedRight + currentKilled
        sequences.append(seq + seqLeft + seqRight)

    # Iterate through all horizontal rectangle boundaries (except borders)
    for i in range( 1, len(y) - 1):

        proposedCut = AxisParallelLine( point = y[i], axis = 'y' ) 
        rectsBelow, rectsAbove = set(), set()    # boundary will split region into above and below subregions
        rectStartingAtBoundary = False
        currentKilled = 0

        #check what the guillotine cut does to each rectangle in the region
        for rect in rects: 

            if lineIntersectsRectangle( proposedCut , rect ):
                # this rectangle is killed
                currentKilled += 1
            elif rect.topRight[1] <= y[i]:
                # rectangle falls below the cut
                rectsBelow.add(rect)
            else:
                rectsAbove.add(rect)
        
            if rect.bottomLeft[1] == y[i]: 
                # the guillotine cut has a boundary starting rectangle
                rectStartingAtBoundary = True

        yBelow = y[ : i+1]
        yAbove = y[i : ] if rectStartingAtBoundary else y[i + 1 : ]

        xBelow = set()
        for rect in rectsBelow:
            xBelow.add(rect.bottomLeft[0])
            xBelow.add(rect.topRight[0])
        xBelow = sorted(list(xBelow))

        xAbove = set()
        for rect in rectsAbove:
            xAbove.add(rect.bottomLeft[0])
            xAbove.add(rect.topRight[0])
        xAbove = sorted(list(xAbove))

        regionBelow = Rectangle( bottomLeft = reg.bottomLeft, topRight = (reg.topRight[0], y[i]) )
        regionAbove = Rectangle( bottomLeft = (reg.bottomLeft[0], y[i]), topRight = reg.topRight )

        seqBelow, killedBelow = findOptimalCuts(rectsBelow, xBelow, yBelow, regionBelow, seq, memo)
        seqAbove, killedAbove = findOptimalCuts(rectsAbove, xAbove, yAbove, regionAbove, seq, memo)

        cuts[len(x) - 2 + i - 1] = killedBelow  + killedAbove + currentKilled
        sequences.append(seq + seqBelow + seqAbove)

    minPtr = 0

    for i in range(numCandidates):
        if cuts[i] < cuts[minPtr]:
            minPtr = i

    # Record Optimal Cut at this level with minimum discarded rectangles
    newLine = AxisParallelLine( point = INF, axis = 'x' )

    if minPtr < len(x) - 2 :
        newLine = AxisParallelLine( point = x[1 + minPtr], axis = 'x' )

    else:
        newLine = AxisParallelLine( point = y[minPtr + 3 - len(x)], axis = 'y' )

    # Add to memo and return
    try:
        memo[reg] = ([reg, newLine] + sequences[minPtr], cuts[minPtr])
    except:
        print(f"minPtr is {minPtr} and sequences list has size {len(sequences)} and cuts has size {len(cuts)}")

    return [reg, newLine] + sequences[minPtr], cuts[minPtr]

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

    ax.add_patch(patches.Rectangle((0,0), max-min, max-min, edgecolor = 'red', fill=False, lw=0.3))

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

rects = [Rectangle( bottomLeft = (0,0), topRight = (1,4) ), Rectangle( bottomLeft = (0,4), topRight = (4,5) ),Rectangle( bottomLeft = (1,0), topRight = (2,3) ), Rectangle( bottomLeft = (2,0), topRight = (5,1) ), Rectangle( bottomLeft = (2,1), topRight = (4,2) ), Rectangle( bottomLeft = (2,2), topRight = (3,3) ),Rectangle( bottomLeft = (3,2), topRight = (4,4) ), Rectangle( bottomLeft = (4,1), topRight = (5,5) ),Rectangle( bottomLeft = (1,3), topRight = (3,4) )]
	
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 	[Rectangle(bottomLeft=(10, 25), topRight=(43, 55)), Rectangle(bottomLeft=(68, 47), topRight=(81, 67)), Rectangle(bottomLeft=(120, 29), topRight=(172, 56)), Rectangle(bottomLeft=(114, 12), topRight=(146, 22)), Rectangle(bottomLeft=(31, 55), topRight=(41, 87)), Rectangle(bottomLeft=(41, 67), topRight=(95, 98)), Rectangle(bottomLeft=(125, 56), topRight=(133, 145)), Rectangle(bottomLeft=(160, 90), topRight=(199, 135)), Rectangle(bottomLeft=(0, 87), topRight=(41, 150)), Rectangle(bottomLeft=(58, 98), topRight=(125, 107))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(25, 0), topRight=(95, 20)), Rectangle(bottomLeft=(60, 25), topRight=(95, 33)), Rectangle(bottomLeft=(95, 2), topRight=(105, 52)), Rectangle(bottomLeft=(105, 0), topRight=(152, 25)), Rectangle(bottomLeft=(0, 45), topRight=(35, 95)), Rectangle(bottomLeft=(35, 20), topRight=(60, 103)), Rectangle(bottomLeft=(87, 45), topRight=(90, 128)), Rectangle(bottomLeft=(139, 25), topRight=(189, 50)), Rectangle(bottomLeft=(0, 134), topRight=(17, 144)), Rectangle(bottomLeft=(35, 103), topRight=(79, 113)), Rectangle(bottomLeft=(90, 68), topRight=(115, 104))]
	
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(26, 0), topRight=(70, 25)), Rectangle(bottomLeft=(41, 25), topRight=(49, 120)), Rectangle(bottomLeft=(70, 0), topRight=(109, 45)), Rectangle(bottomLeft=(109, 0), topRight=(199, 23)), Rectangle(bottomLeft=(50, 69), topRight=(60, 134)), Rectangle(bottomLeft=(60, 45), topRight=(152, 70)), Rectangle(bottomLeft=(126, 70), topRight=(140, 90)), Rectangle(bottomLeft=(152, 23), topRight=(197, 96)), Rectangle(bottomLeft=(8, 152), topRight=(94, 199)), Rectangle(bottomLeft=(60, 90), topRight=(143, 110)), Rectangle(bottomLeft=(63, 110), topRight=(130, 152)), Rectangle(bottomLeft=(130, 142), topRight=(196, 165))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(18, 0), topRight=(85, 50)), Rectangle(bottomLeft=(85, 1), topRight=(173, 25)), Rectangle(bottomLeft=(85, 25), topRight=(130, 60)), Rectangle(bottomLeft=(137, 50), topRight=(142, 70)), Rectangle(bottomLeft=(189, 70), topRight=(199, 164)), Rectangle(bottomLeft=(20, 55), topRight=(39, 98)), Rectangle(bottomLeft=(82, 60), topRight=(92, 107)), Rectangle(bottomLeft=(92, 60), topRight=(101, 88)), Rectangle(bottomLeft=(110, 70), topRight=(147, 90)), Rectangle(bottomLeft=(147, 32), topRight=(165, 73)), Rectangle(bottomLeft=(25, 98), topRight=(47, 125)), Rectangle(bottomLeft=(35, 125), topRight=(95, 160)), Rectangle(bottomLeft=(95, 90), topRight=(124, 182))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(25, 72), topRight=(44, 85)), Rectangle(bottomLeft=(35, 0), topRight=(104, 39)), Rectangle(bottomLeft=(105, 0), topRight=(188, 25)), Rectangle(bottomLeft=(105, 25), topRight=(165, 70)), Rectangle(bottomLeft=(165, 25), topRight=(199, 47)), Rectangle(bottomLeft=(0, 45), topRight=(24, 115)), Rectangle(bottomLeft=(44, 79), topRight=(121, 99)), Rectangle(bottomLeft=(121, 70), topRight=(131, 110)), Rectangle(bottomLeft=(131, 80), topRight=(181, 86)), Rectangle(bottomLeft=(189, 47), topRight=(199, 116)), Rectangle(bottomLeft=(0, 115), topRight=(42, 123)), Rectangle(bottomLeft=(42, 99), topRight=(67, 124)), Rectangle(bottomLeft=(67, 101), topRight=(112, 184)), Rectangle(bottomLeft=(112, 117), topRight=(127, 177))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(0, 0), topRight=(50, 36)), Rectangle(bottomLeft=(76, 24), topRight=(86, 99)), Rectangle(bottomLeft=(70, 0), topRight=(168, 24)), Rectangle(bottomLeft=(135, 24), topRight=(155, 37)), Rectangle(bottomLeft=(162, 24), topRight=(187, 104)), Rectangle(bottomLeft=(10, 36), topRight=(76, 39)), Rectangle(bottomLeft=(43, 95), topRight=(76, 115)), Rectangle(bottomLeft=(86, 33), topRight=(108, 41)), Rectangle(bottomLeft=(105, 65), topRight=(155, 76)), Rectangle(bottomLeft=(155, 24), topRight=(162, 27)), Rectangle(bottomLeft=(15, 90), topRight=(41, 137)), Rectangle(bottomLeft=(60, 115), topRight=(70, 173)), Rectangle(bottomLeft=(70, 118), topRight=(130, 199)), Rectangle(bottomLeft=(130, 90), topRight=(140, 129)), Rectangle(bottomLeft=(140, 104), topRight=(194, 195))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(46, 0), topRight=(115, 25)), Rectangle(bottomLeft=(36, 0), topRight=(46, 50)), Rectangle(bottomLeft=(95, 25), topRight=(105, 66)), Rectangle(bottomLeft=(111, 29), topRight=(199, 65)), Rectangle(bottomLeft=(25, 101), topRight=(74, 156)), Rectangle(bottomLeft=(35, 50), topRight=(72, 93)), Rectangle(bottomLeft=(74, 91), topRight=(96, 122)), Rectangle(bottomLeft=(105, 27), topRight=(107, 34)), Rectangle(bottomLeft=(0, 156), topRight=(77, 179)), Rectangle(bottomLeft=(91, 122), topRight=(101, 177)), Rectangle(bottomLeft=(101, 121), topRight=(157, 148)), Rectangle(bottomLeft=(157, 65), topRight=(175, 133)), Rectangle(bottomLeft=(42, 179), topRight=(85, 199)), Rectangle(bottomLeft=(85, 179), topRight=(92, 199)), Rectangle(bottomLeft=(92, 177), topRight=(129, 181)), Rectangle(bottomLeft=(151, 148), topRight=(177, 199))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(6, 36), topRight=(60, 56)), Rectangle(bottomLeft=(85, 0), topRight=(153, 17)), Rectangle(bottomLeft=(95, 17), topRight=(116, 73)), Rectangle(bottomLeft=(119, 17), topRight=(189, 42)), Rectangle(bottomLeft=(189, 0), topRight=(199, 49)), Rectangle(bottomLeft=(0, 56), topRight=(60, 64)), Rectangle(bottomLeft=(70, 97), topRight=(105, 123)), Rectangle(bottomLeft=(60, 30), topRight=(95, 84)), Rectangle(bottomLeft=(134, 45), topRight=(148, 115)), Rectangle(bottomLeft=(148, 78), topRight=(199, 104)), Rectangle(bottomLeft=(33, 72), topRight=(60, 160)), Rectangle(bottomLeft=(60, 127), topRight=(90, 154)), Rectangle(bottomLeft=(90, 123), topRight=(165, 186)), Rectangle(bottomLeft=(105, 73), topRight=(134, 100)), Rectangle(bottomLeft=(186, 104), topRight=(199, 164)), Rectangle(bottomLeft=(0, 143), topRight=(10, 199)), Rectangle(bottomLeft=(52, 160), topRight=(90, 199))]
plot_rectangles_with_cuts(rects, 0, 200, 0)

rects = 		[Rectangle(bottomLeft=(0, 0), topRight=(13, 80)), Rectangle(bottomLeft=(49, 0), topRight=(104, 25)), Rectangle(bottomLeft=(95, 25), topRight=(105, 45)), Rectangle(bottomLeft=(105, 14), topRight=(157, 34)), Rectangle(bottomLeft=(180, 25), topRight=(199, 98)), Rectangle(bottomLeft=(50, 45), topRight=(52, 85)), Rectangle(bottomLeft=(52, 40), topRight=(90, 90)), Rectangle(bottomLeft=(90, 45), topRight=(164, 70)), Rectangle(bottomLeft=(80, 113), topRight=(95, 179)), Rectangle(bottomLeft=(157, 71), topRight=(180, 96)), Rectangle(bottomLeft=(16, 115), topRight=(32, 135)), Rectangle(bottomLeft=(43, 115), topRight=(80, 135)), Rectangle(bottomLeft=(95, 78), topRight=(120, 136)), Rectangle(bottomLeft=(120, 85), topRight=(140, 184)), Rectangle(bottomLeft=(140, 99), topRight=(190, 173)), Rectangle(bottomLeft=(12, 135), topRight=(75, 160)), Rectangle(bottomLeft=(60, 160), topRight=(70, 199)), Rectangle(bottomLeft=(70, 179), topRight=(108, 199))]
plot_rectangles_with_cuts(rects, 0, 200, 0)