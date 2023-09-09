# Run Observations

**Rectangle generation paradigm is as follows** :

We allow the agent to take N*5 decisions where N = number of rectangles in the generation. In this approach, there are always N rectangles present in the bounded box, the agent simply manipulates their positions and dimensions. 

The agent takes 5 decisions sequentially, the first decision determines which rectangle it is going to work on. The next two decisions make changes to the (x,y) bottom left corner - either by adding or subtracting the normalized action value from this. When an action is picked by the agent, we make sure to readjust the normalized action to the closest action that doesn't cause a collision with the other rectangles or the boundaries of the region. We make the same adjustment to the true action value before saving the action details. The 4th and 5th decisions make changes to the rectangle dimensions. They either add or subtract the normalized action value something from the width and height respectively. Once again the normalized action value and the true action value will both be adjusted to make sure there are no collisions our out-of-bounds.

Interval graphs will be constructed with each change in rectangle parameters and passed into the state. 

What is a normalized action value? Say we have set n_actions = 101, then action values will be normalized by taking the actual action value and subtracting 51 from it so that we have a range of integer values from -50 to 50. The normalized action value will be added to the dimensions or coordinates of the rectangle.

With this approach, we don't let the agent produce anything other than disjoint rectangle sets. Thus, our reward only depends on number of rectangles killed.

**The state representation is as follows** :

The observation space, or state dimensions are 

( region_bound * N * 4 ) + ( 2 * N ) + ( 2 * N * N ) + ( N * 5 )

Let's break this down

1. region_bound * N * 4 - Contains the rectangle set representation, same as the one-hot method we have used before which follows directly from Wagner's template
2. 2 * N - The first N bits represent which rectangles have already been worked on (1) and which ones haven't (0) - This is primarily for the benefit of every 5th decision where the agent chooses the next rectangle to work on. The next N bits represent which rectangle is currently being worked on (1) - one hot encoding
3. 2 * N * N - Interval Graphs of X and Y intervals
4. N * 5 - which decision are we currently working on - one hot encoding

**Observations from the runs** :

To be done

Run 1 - 9 Rectangles

**Another approach - this is currently included** to solving the problem in larger numbers of rectangles could be to keep generating the first iteration again and again until at least one lucky hit is found. We only move on to training once we have a lucky hit.


*Hyperparameters used in run 1*

| Variable Name | Value | Significance |
|--|--|--|
| N | 7 | # Rectangles to generate |
| DECISIONS | N*4 | # 4 coordinates to be decided for each rectangle in generation |
 observation_space |  2*DECISIONS | # 4 coordinates to be decided for each rectangle in generation |
| LEARNING_RATE | 0.0001 | Learning Rate |
| n_sessions | 2000 | Batch Size / # Episodes or Generations per Sessions |
| Percentile | 70 | Top 100-X percent we are learning from after each session |
| super_percentile | 90 | Top 100-X percent of episodes that survive to the next iteration |
| FIRST_LAYER_NEURONS | 256 | - |
| SECOND_LAYER_NEURONS | 128 | - |
| THIRD_LAYER_NEURONS  | 128 | - |
| n_actions | 100 | The action space consists of all integers in [0,100). These will be the rectangle coordinates |
| disjoint_penalty | -2000 | Penalty for not generating a disjoint set of rectangles |  
| reward_scaling | 40 | Scaling factor. We multiply no of rectangles killed by the optimal cut sequence |
| long_episode_penalty | -10000 | Penalty for letting an episode run beyond DECISIONS*100 steps |
