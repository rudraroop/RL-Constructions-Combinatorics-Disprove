#include <bits/stdc++.h>
#include <variant>
using namespace std;

#define INF 10000

using Interval = pair <int, int>;

// struct return_FindOptimalCuts_2 * findOptimalCuts(vector <struct Rectangle *> rects, set <int> x, set <int> y, struct Rectangle * reg, vector <int> seq, map <struct Rectangle *, int> * memo);
// struct return_OptimalCuts_1 * optimalCuts(vector <struct Rectangle *> rects, struct Rectangle * reg);

// this is fine
struct Rectangle
{
    Interval bottomLeft;
    Interval topRight;

    Rectangle(Interval bottomLeft, Interval topRight) : bottomLeft(bottomLeft), topRight(topRight) {}
};
// this is fine
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
    map <Rectangle *, MemoValue *> returnMemo;
};

struct return_FindOptimalCuts_2
{
    vector <GeometryObject> returnVector;
    int returnCount;
    map <Rectangle *, MemoValue *> returnMemo;
};


struct return_OptimalCuts_1 * optimalCuts(vector <struct Rectangle *> rects, struct Rectangle * reg);
struct return_FindOptimalCuts_2 * findOptimalCuts(/*set*/vector <struct Rectangle *> rects, vector <int> x, vector <int> y, struct Rectangle * reg, vector <GeometryObject> seq, map <Rectangle * , MemoValue * > memo);

bool intervalsIntersect(Interval int1, Interval int2)
{
    return ! ((int1.first >= int2.second) || (int2.first >= int1.second));
}
bool lineIntersectsRectangle(struct AxisParallelLine *line, struct Rectangle *rect)
{
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

    map <Rectangle *, MemoValue *> memo;

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

    vector <GeometryObject> seq;

    struct return_FindOptimalCuts_2 * r0 = new struct return_FindOptimalCuts_2();

    r0 = findOptimalCuts(rects, x_vec, y_vec, reg, seq, memo);
    
    returnVariable->flag = true;
    returnVariable->returnVector = r0->returnVector;
    returnVariable->returnCount = r0->returnCount;
    returnVariable->returnMemo = r0->returnMemo;
    return returnVariable;
}

struct return_FindOptimalCuts_2 * findOptimalCuts(/*set*/vector <struct Rectangle *> rects, vector <int> x, vector <int> y, struct Rectangle * reg, vector <GeometryObject> seq, map <Rectangle * , MemoValue * > memo)
{   
    struct return_FindOptimalCuts_2 * returnVariable = new struct return_FindOptimalCuts_2();

    if (rects.size() <= 3)
    {
        returnVariable->returnVector = seq;
        returnVariable->returnCount = 0;
        return returnVariable;
    }

    auto it_memo = memo.find(reg);
    if (it_memo != memo.end())
    {
        MemoValue * value = it_memo->second;
        returnVariable->returnVector = value->objects;
        returnVariable->returnCount = value->state;
        return returnVariable;
    }

    int numCandidates = x.size() + y.size() - 4;
    // int cuts[numCandidates];
    vector <int> cuts;
    for (int i = 0; i < numCandidates; i++)
    {
        cuts.push_back(INF);
    }

    vector <vector <GeometryObject>> sequences;

    for (int i = 1; i < x.size() - 1; i++)
    {
        struct AxisParallelLine * proposedCut = new struct AxisParallelLine(x[i], 0);
        // proposedCut->point = x[i];    
        // proposedCut->axis = 0;

        vector <struct Rectangle *> rectsLeft;
        vector <struct Rectangle *> rectsRight;
        bool rectStartingAtBoundary = false;
        int currentKilled = 0;

        for (int j = 0; j < rects.size(); j++)
        {
            if ( lineIntersectsRectangle( proposedCut, rects[j] ) )
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

        // set <int> xLeft, xRight;
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
            yLeft_set.insert(rects[j]->bottomLeft.second);
            yLeft_set.insert(rects[j]->topRight.second);
        }
        vector <int> yLeft;
        for (const auto& ele : yLeft_set) 
        {
            yLeft.push_back(ele);
        }

        set <int> yRight_set;
        for (int j = 0; j < rectsRight.size(); j++)
        {
            yRight_set.insert(rects[j]->bottomLeft.second);
            yRight_set.insert(rects[j]->topRight.second);
        }
        vector <int> yRight;
        for (const auto& ele : yRight_set)
        {
            yRight.push_back(ele);
        }

        struct Rectangle * regionLeft = new struct Rectangle(reg->bottomLeft, make_pair(x[i], reg->topRight.second));
        // regionLeft->bottomLeft = reg->bottomLeft; 
        // pair <int, int> regionLeftPair = make_pair(x[i], reg->topRight.second);
        // regionLeft->topRight = regionLeftPair;

        struct Rectangle * regionRight = new struct Rectangle(make_pair(x[i], reg->bottomLeft.second), reg->topRight);
        // pair <int, int> regionRightPair = make_pair(x[i], reg->bottomLeft.second);
        // regionRight->bottomLeft = regionRightPair;
        // regionRight->topRight = reg->topRight;

        struct return_FindOptimalCuts_2 * returnLeft = new struct return_FindOptimalCuts_2();
        returnLeft = findOptimalCuts(rectsLeft, xLeft, yLeft, regionLeft, seq, memo);
        vector <GeometryObject> seqLeft = returnLeft->returnVector;
        int killedLeft = returnLeft->returnCount;

        struct return_FindOptimalCuts_2 * returnRight = new struct return_FindOptimalCuts_2();
        returnRight = findOptimalCuts(rectsRight, xRight, yRight, regionRight, seq, memo);
        vector <GeometryObject> seqRight = returnRight->returnVector;
        int killedRight = returnRight->returnCount;
        
        cuts[i-1] = killedLeft + killedRight + currentKilled;

        sequences.push_back(seq);
        sequences.push_back(seqLeft);
        sequences.push_back(seqRight);

        // cout << sequences[0] << endl;
    }

    for (int i = 1; i < y.size() - 1; i++)
    {
        struct AxisParallelLine * proposedCut = new struct AxisParallelLine(y[i], 1);
        // proposedCut->point = y[i];    
        // proposedCut->axis = 1;

        vector <struct Rectangle *> rectsBelow;
        vector <struct Rectangle *> rectsAbove;
        bool rectStartingAtBoundary = false;
        int currentKilled = 0;

        for (int j = 0; j < rects.size(); j++)
        {
            if ( lineIntersectsRectangle( proposedCut, rects[j] ) )
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
            xBelow_set.insert(rects[j]->bottomLeft.first);
            xBelow_set.insert(rects[j]->topRight.first);
        }
        vector <int> xBelow;
        for (const auto& ele : xBelow_set)
        {
            xBelow.push_back(ele);
        }

        set <int> xAbove_set;
        for (int j = 0; j < rectsAbove.size(); j++)
        {
            xAbove_set.insert(rects[j]->bottomLeft.first);
            xAbove_set.insert(rects[j]->topRight.first);
        }
        vector <int> xAbove;
        for (const auto& ele : xAbove_set)
        {
            xAbove.push_back(ele);
        }

        struct Rectangle * regionBelow = new struct Rectangle(reg->bottomLeft, make_pair(reg->topRight.first, y[i]));
        // regionBelow->bottomLeft = reg->bottomLeft; 
        // pair <int, int> regionBelowPair = make_pair(reg->topRight.first, y[i]);
        // regionBelow->topRight = regionBelowPair;

        struct Rectangle * regionAbove = new struct Rectangle(make_pair(reg->bottomLeft.first, y[i]), reg->topRight);
        // pair <int, int> regionAbovePair = make_pair(reg->bottomLeft.first, y[i]);
        // regionAbove->bottomLeft = regionAbovePair;
        // regionAbove->topRight = reg->topRight;

        struct return_FindOptimalCuts_2 * returnBelow = new struct return_FindOptimalCuts_2();
        returnBelow = findOptimalCuts(rectsBelow, xBelow, yBelow, regionBelow, seq, memo);
        vector <GeometryObject> seqBelow = returnBelow->returnVector;
        int killedBelow = returnBelow->returnCount;

        struct return_FindOptimalCuts_2 * returnAbove = new struct return_FindOptimalCuts_2();
        returnAbove = findOptimalCuts(rectsAbove, xAbove, yAbove, regionAbove, seq, memo);
        vector <GeometryObject> seqAbove = returnAbove->returnVector;
        int killedAbove = returnAbove->returnCount;
        
        cuts[x.size() - 2 + i - 1] = killedBelow + killedAbove + currentKilled;

        sequences.push_back(seq);
        sequences.push_back(seqBelow);
        sequences.push_back(seqAbove);

        //cout << sequences[0] << "\n" << sequences[1] << "\n" << sequences[2] << endl;
    }

    int minPtr = 0;

    for (int i = 0; i < numCandidates; i++)
    {
        if (cuts[i] < cuts[minPtr])
        {
            minPtr = i;
        }
    }

    struct AxisParallelLine * newLine = new struct AxisParallelLine(INF, 0);
    // newLine->point = INF;
    // newLine->axis = 0;

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

    try 
    {
        // Using find to check if the key exists in the map
        auto it_memoFinal = memo.find(reg);
        if (it_memoFinal != memo.end()) 
        {
            // Key found, update the value
            // it->second = cuts[minPtr];
            // vector <GeometryObject *> memoObjects;
            vector <GeometryObject> memoObjects;
            memoObjects.push_back(reg);
            memoObjects.push_back(newLine);
            for (int j = 0; j < sequences[minPtr].size(); j++)
            {
                memoObjects.push_back(sequences[minPtr][j]);
            }
            it_memoFinal->second->objects = memoObjects;
            it_memoFinal->second->state = cuts[minPtr];
        } 
    } 
    catch (const std::exception& e) 
    {
        std::cout << "Exception caught: " << e.what() << "\n" << "minPtr is " << minPtr << " and sequences list has size " << sequences.size() << " and cuts has size " << cuts.size() << std::endl;
    }
    
    vector <GeometryObject> returnObjects;
    returnObjects.push_back(reg);
    returnObjects.push_back(newLine);
    for (int k = 0; k < sequences[minPtr].size(); k++)
    {
        returnObjects.push_back(sequences[minPtr][k]);
    }
    returnVariable->returnVector = returnObjects;
    returnVariable->returnCount = cuts[minPtr];  
    returnVariable->returnMemo = memo;
    return returnVariable;
}

struct Printer
{
    void operator()(Rectangle * r) const {cout << "Rectangle = Bottom Left :" << r->bottomLeft.first << ", " << r->bottomLeft.second << "\nTop Right : " << r->topRight.first << ", " << r->topRight.second << endl;}
    void operator()(AxisParallelLine * apl) const {cout << "AxisParallelLine = Point : " << apl->point << "\nAxis : " << apl->axis << endl;}
};

int main() 
{
    Rectangle* boundedRegion = new Rectangle(Interval(0, 0), Interval(100, 100));

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

    // test 3 [plots/run_3/displayed_generations.txt]
    testRectangles.push_back(new Rectangle(Interval(0, 55), Interval(35, 85)));
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


    // Calling the optimalCuts function
    struct return_OptimalCuts_1 * returning = new struct return_OptimalCuts_1();
    returning = optimalCuts(testRectangles, boundedRegion);

    // Printing the results
    cout << "flag = " << returning->flag << endl;
    for (const auto& val : returning->returnVector)
    {
        visit(Printer{}, val);
    }
    // cout << "vector[0] = " << returning->returnVector[0] << endl;
    cout << "count = " << returning->returnCount << endl;

    // Freeing input rectangles
    for (auto rect : testRectangles) 
    {
        delete rect;
    }

    // Freeing Bounded Region
    delete boundedRegion;

    // Freeing Memoized Objects
    for (auto& entry : returning->returnMemo)
    {
        delete entry.second;
    }
    returning->returnMemo.clear();

    //
    // for (auto entry : returning->returnVector)
    // {
    //     delete entry;
    // }

    // delete &returning->returnVector;
    // delete &returning->returnMemo;

    return 0;
}
