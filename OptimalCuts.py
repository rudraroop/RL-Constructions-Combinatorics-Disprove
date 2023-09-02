from collections import namedtuple

from plotGenerations import plot_rectangles

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

# validate optimal cuts algorithm - 9 rectangle pinwheel

rectangles = []
rectangles.append(Rectangle( bottomLeft = (0,0), topRight = (1,4) ))
rectangles.append(Rectangle( bottomLeft = (0,4), topRight = (4,5) ))
rectangles.append(Rectangle( bottomLeft = (1,0), topRight = (2,3) ))
rectangles.append(Rectangle( bottomLeft = (2,0), topRight = (5,1) ))
rectangles.append(Rectangle( bottomLeft = (2,1), topRight = (4,2) ))
rectangles.append(Rectangle( bottomLeft = (2,2), topRight = (3,3) ))
rectangles.append(Rectangle( bottomLeft = (3,2), topRight = (4,4) ))
rectangles.append(Rectangle( bottomLeft = (4,1), topRight = (5,5) ))
rectangles.append(Rectangle( bottomLeft = (1,3), topRight = (3,4) ))
plot_rectangles(rectangles, 2, 0, 0, 6)

#print(optimalCuts(rectangles, Rectangle(bottomLeft=(0,0), topRight=(100,100))))