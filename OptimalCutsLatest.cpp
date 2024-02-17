#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <variant>
#include <memory>
// #include <chrono>

using namespace std;

#define INF 10000

int NewCounter_Rectangle;
int NewCounter_AxisParallelLine;
int NewCounter_return_OptimalCuts_1;
int NewCounter_return_FindOptimalCuts_2;

int stackDepth = 0;
int maxStackDepth = 0;
int functionInvokvedCounter_findOptimus;
int lineIntersectsRectangle_func_call_cnt;
int isDisjoint_func_call_cnt;
int findOptimalCuts_return1_cnt;
int findOptimalCuts_return2_cnt;


void printNewCounters ()
{
    cout << endl;
    cout << "findOptimalCuts_return1_cnt:  " << findOptimalCuts_return1_cnt << endl;
    cout << "findOptimalCuts_return2_cnt:  " << findOptimalCuts_return2_cnt << endl;
    cout << "isDisjoint_func_call_cnt:  " << isDisjoint_func_call_cnt << endl;
    cout << "lineIntersectsRectangle_func_call_cnt:  " << lineIntersectsRectangle_func_call_cnt << endl;
    cout << "Stack Depth:               " << stackDepth << endl;
    cout << "Max Stack Depth:           " << maxStackDepth << endl;
    cout << "findOptimus Func Ct:       " << functionInvokvedCounter_findOptimus << endl;
    cout << "Rectangle:                 " << NewCounter_Rectangle << endl;
    cout << "Axis Line:                 " << NewCounter_AxisParallelLine << endl;
    cout << "return_OptimalCuts_1:      " << NewCounter_return_OptimalCuts_1 << endl;
    cout << "return_FindOptimalCuts_2:  " << NewCounter_return_FindOptimalCuts_2 << endl;
    cout << endl;
}

using Interval = pair <int, int>;

struct Rectangle
{
    Interval bottomLeft;
    Interval topRight;

    Rectangle(Interval bottomLeft, Interval topRight) : bottomLeft(bottomLeft), topRight(topRight) {}
};
struct AxisParallelLine
{
    int point;
    int axis;      
    // x = 0, y = 1;

    AxisParallelLine(int p, int a) : point(p), axis(a) {}
};

using GeometryObject = std::variant<Rectangle *, AxisParallelLine *>;

struct MemoValue 
{
    vector <GeometryObject> objects;
    int state;
};

struct return_OptimalCuts_1
{
    bool flag;
    vector <GeometryObject> returnVector;
    int returnCount;
};

struct return_FindOptimalCuts_2
{
    vector <GeometryObject> returnVector;
    int returnCount;
};


struct return_OptimalCuts_1 * optimalCuts(vector <struct Rectangle *> rects, struct Rectangle * reg);
struct return_FindOptimalCuts_2 findOptimalCuts(/*set*/vector <struct Rectangle *> rects, vector <int> x, vector <int> y, struct Rectangle * reg, vector <GeometryObject> seq, unordered_map <Rectangle * , MemoValue * >& memo);

struct Printer
{
    void operator()(Rectangle * r) const {cout << "Rectangle(bottomLeft=(" << r->bottomLeft.first << ", " << r->bottomLeft.second << "), Top Right=(" << r->topRight.first << ", " << r->topRight.second << ")\n";}
    void operator()(AxisParallelLine * apl) const {cout << "AxisParallelLine(point=" << apl->point << ", axis=" << apl->axis << ")\n";}
};

struct RectanglePtrCompare {
    bool operator()(Rectangle* lhs, Rectangle* rhs) const {
		if ((lhs->bottomLeft == rhs->bottomLeft) 
			&& (lhs->topRight == rhs->topRight)) {
			return true;
		}
		else {
			return false;
		}
    }
};

void 
printGeometry (int printHeader, string addStr, GeometryObject geoMetryObject)
{
	if (printHeader)
    	cout << addStr<< ": " << endl;
    visit(Printer{}, geoMetryObject);
	cout << endl;
}

void 
printGeometryVector (int printHeader, string addStr, vector<GeometryObject> geometryObjectVector)
{
    if (geometryObjectVector.empty())
    {
        cout << addStr << ": Empty" << endl;
        return;
    }

    for (const auto& element : geometryObjectVector)
    {
        printGeometry (printHeader, addStr, element);
    }
    printf ("\n");
}

void 
printGeometryVectorOfVector (string addStr, vector<vector<GeometryObject>> geometryObjectVectorOfVector)
{
    if (geometryObjectVectorOfVector.empty())
    {
        cout << addStr << ": []" << endl;
        return;
    }

    cout << addStr << ": " << endl;

    for (const auto& geometryObjectVector : geometryObjectVectorOfVector)
    {
        if (geometryObjectVector.empty())
		{
        	cout << ": []" << endl;
			continue;
		}
        printGeometryVector (0, addStr, geometryObjectVector);
        printf ("\n");
    }
    cout << endl;
}


void
printIntVector (string addStr, vector<int> intVector)
{
    cout << addStr << ": ";
    for (const auto& element : intVector) 
    {
        printf ("%d, ", element);
    }
    printf ("\n");
}

void
printRectangle (string addStr, Rectangle *rect)
{
    cout << addStr << ": Rectangle (";
    printf ("bottonLeft: (%d, %d), ", rect->bottomLeft.first, rect->bottomLeft.second);
    printf ("topRight: (%d, %d)", rect->topRight.first, rect->topRight.second);
    printf (")\n");
}

void
printAxis (string addStr, AxisParallelLine *axis)
{
    cout << addStr << ": AxisLine (" << axis->point << ", " << axis->axis << " )" << endl;
}

void
printRectangleVector (string addStr, vector <struct Rectangle *> rects)
{
    for (const auto& rect : rects)
    {
        printRectangle (addStr, rect);
    }
}

void
printRectangleSet (string addStr, set <struct Rectangle *> rects)
{
  for (struct Rectangle * element : rects) 
  {
        printRectangle (addStr, element);
  }    
}

void
printMemoValue (string addStr, Rectangle *reg, MemoValue *memoValue)
{
    printRectangle("key: ", reg);
    if (memoValue)
        printGeometryVector (1, "value: ", memoValue->objects);
}


void
printMemoMap (string addStr, unordered_map <Rectangle *, MemoValue *> memoMap)
{
    if (memoMap.empty()) 
    {
        cout << addStr << ": Empty" << endl;
        return;
    }

    cout << addStr << ": " << endl;
    for (const auto& [key, value] : memoMap) 
    {
		cout << "key ptr: " << key << endl;
        printMemoValue ("", key, value);
    }
}


bool intervalsIntersect(Interval int1, Interval int2)
{
    return ! ((int1.first >= int2.second) || (int2.first >= int1.second));
}

bool lineIntersectsRectangle(struct AxisParallelLine *line, struct Rectangle *rect)
{
    lineIntersectsRectangle_func_call_cnt++;

    if (line->axis == 0)
    {
        pair <int, int> p1 = make_pair(rect->bottomLeft.first, rect->topRight.first);   // maybe change the parameters of intervalsIntersect
        pair <int, int> p2 = make_pair(line->point, line->point);                       // to accomodate more instead of wasting space by creating pairs?
        return intervalsIntersect(p1, p2);
    }
    else
    {
        pair <int, int> p1 = make_pair(rect->bottomLeft.second, rect->topRight.second);
        pair <int, int> p2 = make_pair(line->point, line->point);
        return intervalsIntersect(p1, p2);
    }
}
bool isDisjoint(struct Rectangle *rect1, struct Rectangle * rect2)
{
    isDisjoint_func_call_cnt++;    
    pair <int, int> xInterval1 = make_pair(rect1->bottomLeft.first, rect1->topRight.first);
    pair <int, int> xInterval2 = make_pair(rect2->bottomLeft.first, rect2->topRight.first);
    pair <int, int> yInterval1 = make_pair(rect1->bottomLeft.second, rect1->topRight.second);
    pair <int, int> yInterval2 = make_pair(rect2->bottomLeft.second, rect2->topRight.second);
    return ! ((intervalsIntersect(xInterval1, xInterval2)) && (intervalsIntersect(yInterval1, yInterval2)));
}

struct return_OptimalCuts_1 * optimalCuts(vector <struct Rectangle *> rects, struct Rectangle * reg)
{
    struct return_OptimalCuts_1 * returnVariable = new struct return_OptimalCuts_1(); 

    for (int i = 0; i < rects.size(); i++)
    {
        for (int j = i+1; j < rects.size(); j++)
        {
            if ( ! (isDisjoint(rects[i], rects[j])) )
            {
                returnVariable->flag = false;
                vector <GeometryObject> returnV;
                returnVariable->returnVector = returnV;
                returnVariable->returnCount = 0;
                return returnVariable;
            }
        }
    }

    unordered_map <Rectangle *, MemoValue *> memo;

    set <int> x_set;
    set <int> y_set;

    for (auto rectangle = rects.begin(); rectangle != rects.end(); ++rectangle)
    {
        x_set.insert((*rectangle)->bottomLeft.first);
        x_set.insert((*rectangle)->topRight.first);
        y_set.insert((*rectangle)->bottomLeft.second);
        y_set.insert((*rectangle)->topRight.second);
    }
    
    vector <int> x_vec; 
    vector <int> y_vec;
    for (const auto& x : x_set) 
    {
        x_vec.push_back(x);
    }
    for (const auto& y : y_set)
    {
        y_vec.push_back(y);
    }
    std::sort(x_vec.begin(), x_vec.end());
    std::sort(y_vec.begin(), y_vec.end());

    vector <GeometryObject> seq;

    struct return_FindOptimalCuts_2 r0;

    r0 = findOptimalCuts(rects, x_vec, y_vec, reg, seq, memo);
    
    returnVariable->flag = true;
    returnVariable->returnVector = r0.returnVector;
    returnVariable->returnCount = r0.returnCount;
    return returnVariable;
}

void
updateStackDepth (int entry)
{
    if (entry) {
        stackDepth++;
        if (stackDepth >= maxStackDepth) {
            maxStackDepth = stackDepth;
        }
        return;
    }

    stackDepth--;
}

// bit slow, O(n) at the minute.
// try to optimize using binary search 
MemoValue *
searchInMemoMap (Rectangle *reg, unordered_map <Rectangle * , MemoValue *>& memoMap)
{
    Interval bottomLeft;
    Interval topRight;
    for (const auto& [key, value] : memoMap) 
    {
		if ((key->bottomLeft == reg->bottomLeft) && (key->topRight == reg->topRight))
		{
			return value;
		}
    }
	return NULL;

}

struct return_FindOptimalCuts_2 findOptimalCuts (vector <struct Rectangle *> rects, 
                                                vector <int> x, vector <int> y, 
                                                struct Rectangle * reg, vector <GeometryObject> seq, 
                                                unordered_map <Rectangle * , MemoValue *>& memo)
{   
    updateStackDepth (1);
    functionInvokvedCounter_findOptimus++;
    struct return_FindOptimalCuts_2 returnVariable;

    // printf ("\n");
    // printf ("Begin, stackDepth: %d, cnt #: %d\n", stackDepth, functionInvokvedCounter_findOptimus);
    // printf ("rects.size: %lu\n", rects.size());
    // printRectangleVector ("rects", rects);
    // printRectangle ("reg", reg);
	// cout << "reg ptr: " << reg << endl;
    // printIntVector ("x", x);
    // printIntVector ("y", y);
    // printf ("seq.size: %lu\n", seq.size());
    // printGeometryVector(1, "seq", seq);
    // printf ("memo.size: %lu\n", memo.size());
    // printMemoMap ("memo:", memo);
    // printf ("\n");

    if (rects.size() <= 3)
    {
        returnVariable.returnVector = seq;
        returnVariable.returnCount = 0;
        updateStackDepth (0);
        findOptimalCuts_return1_cnt++;
        // printf ("Returning, Size LEQ 3, stackDepth: %d, cnt #: %d\n", stackDepth, functionInvokvedCounter_findOptimus);
        return returnVariable;
    }

	//Orig
    //auto it_memo = memo.find(reg);
    //if (it_memo != memo.end())
	MemoValue *memoValue = searchInMemoMap (reg, memo);
	if (memoValue)
    {
        //MemoValue * value = it_memo->second;
        returnVariable.returnVector = memoValue->objects;
        returnVariable.returnCount = memoValue->state;
        updateStackDepth (0);
        findOptimalCuts_return2_cnt++;
        // printf ("Returning, Found in memo, stackDepth: %d, cnt #: %d\n", stackDepth, functionInvokvedCounter_findOptimus);
        // printMemoValue("mem[reg]: ", reg, memoValue);
        return returnVariable;
    }

    int numCandidates = x.size() + y.size() - 4;

    //Orig
    // vector <int> cuts;
    // for (int i = 0; i < numCandidates; i++)
    // {
    //     cuts.push_back(INF);
    // }
    std::vector<int> cuts(numCandidates, INF);

    vector <vector <GeometryObject>> sequences{};

    for (int i = 1; i < x.size() - 1; i++)
    {
		// printf ("x-for-loop, i: %d, begin\n", i);
        struct AxisParallelLine proposedCut2(x[i], 0);

        vector <struct Rectangle *> rectsLeft;
        vector <struct Rectangle *> rectsRight;

        bool rectStartingAtBoundary = false;
        int currentKilled = 0;

        for (int j = 0; j < rects.size(); j++)
        {
            if ( lineIntersectsRectangle(&proposedCut2, rects[j] ) )
            {   
                currentKilled++;
            }
            else if (rects[j]->topRight.first <= x[i])
            {
                rectsLeft.push_back(rects[j]);
            }
            else
            {
                rectsRight.push_back(rects[j]);
            }

            if (rects[j]->bottomLeft.first == x[i])
            {
                rectStartingAtBoundary = true;
            }
        } 

        vector <int> xLeft;
        vector <int> xRight;
        for (int j = 0; j < i+1; j++)
        {
            xLeft.push_back(x[j]);
        }

        if (rectStartingAtBoundary)
        {
            for (int j = i; j < x.size(); j++)
            {
                xRight.push_back(x[j]);
            }
        }
        else
        {
            for (int j = i+1; j < x.size(); j++)
            {
                xRight.push_back(x[j]);
            }
        }

        set <int> yLeft_set;
        for (int j = 0; j < rectsLeft.size(); j++)
        {
            yLeft_set.insert(rectsLeft[j]->bottomLeft.second);
            yLeft_set.insert(rectsLeft[j]->topRight.second);
        }

        vector <int> yLeft;
        for (const auto& ele : yLeft_set) 
        {
            yLeft.push_back(ele);
        }
        std::sort(yLeft.begin(), yLeft.end());

        set <int> yRight_set;
        for (int j = 0; j < rectsRight.size(); j++)
        {
            yRight_set.insert(rectsRight[j]->bottomLeft.second);
            yRight_set.insert(rectsRight[j]->topRight.second);
        }

        vector <int> yRight;
        for (const auto& ele : yRight_set)
        {
            yRight.push_back(ele);
        }
        std::sort(yRight.begin(), yRight.end());


        struct Rectangle *regionLeft = new struct Rectangle(reg->bottomLeft, make_pair(x[i], reg->topRight.second));
        struct Rectangle *regionRight = new struct Rectangle(make_pair(x[i], reg->bottomLeft.second), reg->topRight);

        struct return_FindOptimalCuts_2 returnLeft;
        returnLeft = findOptimalCuts(rectsLeft, xLeft, yLeft, regionLeft, seq, memo);
        vector <GeometryObject> seqLeft = returnLeft.returnVector;
        int killedLeft = returnLeft.returnCount;

#if 0
        printf ("return value after 1st-1st, cnt #: %d\n", functionInvokvedCounter_findOptimus);
        printGeometryVector(1, "seqLeft:", seqLeft);
        printf ("killedLeft: %d\n", killedLeft);
		printMemoMap ("updated memo after 1st-1st", memo);
#endif

        struct return_FindOptimalCuts_2 returnRight;
        returnRight = findOptimalCuts(rectsRight, xRight, yRight, regionRight, seq, memo);
        vector <GeometryObject> seqRight = returnRight.returnVector;
        int killedRight = returnRight.returnCount;

#if 0
        printf ("return value after 1st-2nd, cnt #: %d\n", functionInvokvedCounter_findOptimus);
        printGeometryVector(1, "seqRight:", seqRight);
        printf ("killedRight: %d\n", killedRight);
		printMemoMap ("updated memo after 1st-2nd", memo);
#endif

        cuts[i-1] = killedLeft + killedRight + currentKilled;

		vector<GeometryObject> sumSeq{};
		// Pre-allocate space for efficiency
    	sumSeq.reserve(seq.size() + seqLeft.size() + seqRight.size()); 
    	sumSeq.insert(sumSeq.end(), seq.begin(), seq.end());
    	sumSeq.insert(sumSeq.end(), seqLeft.begin(), seqLeft.end());
    	sumSeq.insert(sumSeq.end(), seqRight.begin(), seqRight.end());

        sequences.push_back(sumSeq);

#if 0
        printGeometryVector (1, "x-for-loop seq", seq);
        printGeometryVector (1, "x-for-loop seqLeft", seqLeft);
        printGeometryVector (1, "x-for-loop seqRight", seqRight);
    	printGeometryVectorOfVector("sequences", sequences);
#endif

		// printf ("x-for-loop, i: %d, end\n", i);
    }

#if 0
    printGeometryVectorOfVector("sequences dump after x-for-loop", sequences);
#endif

    for (int i = 1; i < y.size() - 1; i++)
    {
		// printf ("y-for-loop, i: %d, begin\n", i);
        struct AxisParallelLine proposedCut2(y[i], 1);

        vector <struct Rectangle *> rectsBelow;
        vector <struct Rectangle *> rectsAbove;
        bool rectStartingAtBoundary = false;
        int currentKilled = 0;

        for (int j = 0; j < rects.size(); j++)
        {
            if ( lineIntersectsRectangle(&proposedCut2, rects[j] ) )
            {   
                currentKilled++;
            }
            else if (rects[j]->topRight.second <= y[i])
            {
                rectsBelow.push_back(rects[j]);
            }
            else
            {
                rectsAbove.push_back(rects[j]);
            }

            if (rects[j]->bottomLeft.second == y[i])
            {
                rectStartingAtBoundary = true;
            }
        } 

        // set <int> yBelow, yAbove;
        vector <int> yBelow;
        vector <int> yAbove;
        for (int j = 0; j < i+1; j++)
        {
            yBelow.push_back(y[j]);
        }
        if (rectStartingAtBoundary)
        {
            for (int j = i; j < y.size(); j++)
            {
                yAbove.push_back(y[j]);
            }
        }
        else
        {
            for (int j = i+1; j < y.size(); j++)
            {
                yAbove.push_back(y[j]);
            }
        }

        set <int> xBelow_set;
        for (int j = 0; j < rectsBelow.size(); j++)
        {
            xBelow_set.insert(rectsBelow[j]->bottomLeft.first);
            xBelow_set.insert(rectsBelow[j]->topRight.first);
        }
        vector <int> xBelow;
        for (const auto& ele : xBelow_set)
        {
            xBelow.push_back(ele);
        }
        std::sort(xBelow.begin(), xBelow.end());
        

        set <int> xAbove_set;
        for (int j = 0; j < rectsAbove.size(); j++)
        {
            xAbove_set.insert(rectsAbove[j]->bottomLeft.first);
            xAbove_set.insert(rectsAbove[j]->topRight.first);
        }
        vector <int> xAbove;
        for (const auto& ele : xAbove_set)
        {
            xAbove.push_back(ele);
        }
        std::sort(xAbove.begin(), xAbove.end());

        //struct Rectangle regionBelow(reg->bottomLeft, make_pair(reg->topRight.first, y[i]));
        //struct Rectangle regionAbove(make_pair(reg->bottomLeft.first, y[i]), reg->topRight);
        struct Rectangle *regionBelow = new struct Rectangle(reg->bottomLeft, make_pair(reg->topRight.first, y[i]));
        struct Rectangle *regionAbove = new struct Rectangle(make_pair(reg->bottomLeft.first, y[i]), reg->topRight);

        struct return_FindOptimalCuts_2 returnBelow;
        returnBelow = findOptimalCuts(rectsBelow, xBelow, yBelow, regionBelow, seq, memo);
        vector <GeometryObject> seqBelow = returnBelow.returnVector;
        int killedBelow = returnBelow.returnCount;

#if 0
        printf ("return value after 2nd-1st, cnt #: %d\n", functionInvokvedCounter_findOptimus);
        printGeometryVector(1, "seqBelow:", seqBelow);
        printf ("killedBelow: %d\n", killedBelow);
		printMemoMap ("updated memo after 2nd-1st", memo);
#endif


        struct return_FindOptimalCuts_2 returnAbove;
        returnAbove = findOptimalCuts(rectsAbove, xAbove, yAbove, regionAbove, seq, memo);
        vector <GeometryObject> seqAbove = returnAbove.returnVector;
        int killedAbove = returnAbove.returnCount;

#if 0
        printf ("return value after 2nd-2nd, cnt #: %d\n", functionInvokvedCounter_findOptimus);
        printGeometryVector(1, "seqAbove:", seqAbove);
        printf ("killedAbove: %d\n", killedAbove);
		printMemoMap ("updated memo after 2nd-2nd", memo);
#endif


        cuts[x.size() - 2 + i - 1] = killedBelow + killedAbove + currentKilled;

		// Pre-allocate space for efficiency
		vector<GeometryObject> sumSeq{};
    	sumSeq.reserve(seq.size() + seqBelow.size() + seqAbove.size()); 
    	sumSeq.insert(sumSeq.end(), seq.begin(), seq.end());
    	sumSeq.insert(sumSeq.end(), seqBelow.begin(), seqBelow.end());
    	sumSeq.insert(sumSeq.end(), seqAbove.begin(), seqAbove.end());

        sequences.push_back(sumSeq);

#if 0
        printGeometryVector (1, "y-for-loop seq", seq);
        printGeometryVector (1, "y-for-loop seqBelow", seqBelow);
        printGeometryVector (1, "y-for-loop seqAbove", seqAbove);
    	printGeometryVectorOfVector("sequences", sequences);
#endif

		// printf ("y-for-loop, i: %d, end\n", i);
    }

#if 0
	printGeometryVectorOfVector("sequences dump after y-for-loop", sequences);
#endif

    int minPtr = 0;

    for (int i = 0; i < numCandidates; i++)
    {
        if (cuts[i] < cuts[minPtr])
        {
            minPtr = i;
        }
    }

    struct AxisParallelLine *newLine = new struct AxisParallelLine(INF, 0);

    if (minPtr < (x.size() - 2))
    {
        newLine->point = x[1 + minPtr];
        newLine->axis = 0;
    }
    else
    {
        newLine->point = y[minPtr + 3 - x.size()];
        newLine->axis = 1;
    }

    vector <GeometryObject> memoObjects{};

    try 
    {
        memoObjects.clear();
        memoObjects.push_back(reg);
        memoObjects.push_back(newLine);
        if ((sequences.size() != 0) && (sequences.size() > minPtr)) {
            for (int j = 0; j < sequences[minPtr].size(); j++)
            {
                memoObjects.push_back(sequences[minPtr][j]);
            }
        }
        MemoValue *memoValue =  new MemoValue();
        memoValue->objects = memoObjects;
        memoValue->state = cuts[minPtr];

#if 0
		printf ("minPtr: %d\n", minPtr);
		printf ("memo dump before, cnt: %d\n", functionInvokvedCounter_findOptimus);
        printMemoMap("memo dump before", memo);
		printGeometryVector (1, "new addition", memoObjects);
#endif

        memo.insert({reg, memoValue});

#if 0
		printf ("memo dump after, cnt: %d\n", functionInvokvedCounter_findOptimus);
        printMemoMap("memo dump after", memo);
#endif

    } 
    catch (const std::exception& e) 
    {
        std::cout << "Exception caught: " << e.what() << "\n"
                << "minPtr is " << minPtr 
                << " and sequences list has size " << sequences.size() 
                << " and cuts has size " << cuts.size() << std::endl;
    }
    
    returnVariable.returnVector = memoObjects;
    returnVariable.returnCount = cuts[minPtr];
    updateStackDepth (0);
    // printf ("Returning, from the good place, cnt: %d\n", functionInvokvedCounter_findOptimus);
    return returnVariable;
}


int main() 
{
    // auto start = std::chrono::high_resolution_clock::now();

    Rectangle* boundedRegion = new struct Rectangle(Interval(0, 0), Interval(100, 100));

    vector <Rectangle*> testRectangles;

    // test 1
    // testRectangles.push_back(new Rectangle(Interval(0,0), Interval(1,4)));
    // testRectangles.push_back(new Rectangle(Interval(0,4), Interval(4,5)));
    // testRectangles.push_back(new Rectangle(Interval(1,0), Interval(2,3)));
    // testRectangles.push_back(new Rectangle(Interval(2,0), Interval(5,1)));
    // testRectangles.push_back(new Rectangle(Interval(2,1), Interval(4,2)));
    // testRectangles.push_back(new Rectangle(Interval(2,2), Interval(3,3)));
    // testRectangles.push_back(new Rectangle(Interval(3,2), Interval(4,4)));
    // testRectangles.push_back(new Rectangle(Interval(4,1), Interval(5,5)));
    // testRectangles.push_back(new Rectangle(Interval(1,3), Interval(3,4)));

    // test 2 [bottom of plotGenerations.py]
    // [Rectangle(bottomLeft=(25, 72), topRight=(44, 85)), Rectangle(bottomLeft=(35, 0), topRight=(104, 39)), Rectangle(bottomLeft=(105, 0), topRight=(188, 25)), Rectangle(bottomLeft=(105, 25), topRight=(165, 70)), Rectangle(bottomLeft=(165, 25), topRight=(199, 47)), Rectangle(bottomLeft=(0, 45), topRight=(24, 115)), Rectangle(bottomLeft=(44, 79), topRight=(121, 99)), Rectangle(bottomLeft=(121, 70), topRight=(131, 110)), Rectangle(bottomLeft=(131, 80), topRight=(181, 86)), Rectangle(bottomLeft=(189, 47), topRight=(199, 116)), Rectangle(bottomLeft=(0, 115), topRight=(42, 123)), Rectangle(bottomLeft=(42, 99), topRight=(67, 124)), Rectangle(bottomLeft=(67, 101), topRight=(112, 184)), Rectangle(bottomLeft=(112, 117), topRight=(127, 177))]
    // testRectangles.push_back(new Rectangle(Interval(25, 72), Interval(44, 85))); // Example based on the given input
    // testRectangles.push_back(new Rectangle(Interval(35, 0), Interval(104, 39)));
    // testRectangles.push_back(new Rectangle(Interval(105, 0), Interval(188, 25)));
    // testRectangles.push_back(new Rectangle(Interval(105, 25), Interval(165, 70)));
    // testRectangles.push_back(new Rectangle(Interval(165, 25), Interval(199, 47)));
    // testRectangles.push_back(new Rectangle(Interval(0, 45), Interval(24, 115)));
    // testRectangles.push_back(new Rectangle(Interval(44, 79), Interval(121, 99)));
    // testRectangles.push_back(new Rectangle(Interval(121, 70), Interval(131, 110)));
    // testRectangles.push_back(new Rectangle(Interval(131, 80), Interval(181, 86)));
    // testRectangles.push_back(new Rectangle(Interval(189, 47), Interval(199, 116)));
    // testRectangles.push_back(new Rectangle(Interval(0, 115), Interval(42, 123)));
    // testRectangles.push_back(new Rectangle(Interval(42, 99), Interval(67, 124)));
    // testRectangles.push_back(new Rectangle(Interval(67, 101), Interval(112, 184)));
    // testRectangles.push_back(new Rectangle(Interval(112, 117), Interval(127, 177)));


    // test 3
    testRectangles.push_back(new struct Rectangle(Interval(0, 55), Interval(35, 85)));
    testRectangles.push_back(new Rectangle(Interval(35, 0), Interval(77, 62)));
    testRectangles.push_back(new Rectangle(Interval(89, 0), Interval(101, 31)));
    testRectangles.push_back(new Rectangle(Interval(127, 20), Interval(159, 38)));
    testRectangles.push_back(new Rectangle(Interval(159, 26), Interval(199, 91)));
    testRectangles.push_back(new Rectangle(Interval(26, 85), Interval(36, 102)));
    testRectangles.push_back(new Rectangle(Interval(36, 111), Interval(73, 119)));
    testRectangles.push_back(new Rectangle(Interval(68, 85), Interval(140, 89)));
    testRectangles.push_back(new Rectangle(Interval(130, 89), Interval(140, 121)));
    testRectangles.push_back(new Rectangle(Interval(140, 91), Interval(199, 131)));
    testRectangles.push_back(new Rectangle(Interval(0, 119), Interval(38, 160)));
    testRectangles.push_back(new Rectangle(Interval(44, 62), Interval(45, 111)));
    testRectangles.push_back(new Rectangle(Interval(84, 130), Interval(100, 179)));
    testRectangles.push_back(new Rectangle(Interval(106, 103), Interval(130, 162)));
    testRectangles.push_back(new Rectangle(Interval(181, 131), Interval(199, 162)));
    testRectangles.push_back(new Rectangle(Interval(34, 160), Interval(58, 199)));
    testRectangles.push_back(new Rectangle(Interval(58, 169), Interval(69, 189)));
    testRectangles.push_back(new Rectangle(Interval(69, 179), Interval(114, 199)));
    testRectangles.push_back(new Rectangle(Interval(114, 162), Interval(136, 199)));


    NewCounter_Rectangle += 19;


    // Allocate memory for inputRectangle and initialize it
    Rectangle* inputRectangle = new Rectangle(Interval(0, 0), Interval(100, 100));
    NewCounter_Rectangle++;


    // Call the optimalCuts function
    struct return_OptimalCuts_1 * returning = new struct return_OptimalCuts_1();
    NewCounter_return_OptimalCuts_1++;

    // Returning and Printing
    returning = optimalCuts(testRectangles, boundedRegion);
#if 0
    // cout << "flag = " << returning->flag << endl;
    // for (const auto& val : returning->returnVector)
    // {
    //     visit(Printer{}, val);
    // }
    // cout << "count = " << returning->returnCount << endl;
#endif

    // Deleting
    for (auto rect : testRectangles) 
    {
        delete rect;
    }
    delete boundedRegion;

    delete returning;
    NewCounter_return_OptimalCuts_1--;

    NewCounter_Rectangle -= 19;
    NewCounter_Rectangle--;

    // printNewCounters();
    delete (inputRectangle);

    // auto finish = std::chrono::high_resolution_clock::now();
    // chrono::duration<double> elapsed = finish - start;
    // cout << "Elapsed time : " << elapsed.count() << " seconds" << endl;

    return 0;
}
