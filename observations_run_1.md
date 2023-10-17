# Run Observations

**Rectangle generation paradigm is as follows** :

We allow the agent to take N*5 decisions where N = number of rectangles in the generation. In this approach, there are always N rectangles present in the bounded box, the agent simply manipulates their positions and dimensions. 

The agent takes 5 decisions sequentially, the first decision determines which rectangle it is going to work on. The next two decisions make changes to the position of the rectangle (x1,y1) bottom left corner - either by adding or subtracting the normalized action value from this. When an action is picked by the agent, we make sure to readjust the normalized action to the closest action that doesn't cause a collision with the other rectangles or the boundaries of the region. We make the same adjustment to the true action value before saving the action details. The 4th and 5th decisions make changes to the rectangle dimensions. They either add or subtract the normalized action value something from the width and height respectively. Once again the normalized action value and the true action value will both be adjusted to make sure there are no collisions our out-of-bounds.

Interval graphs will be constructed with each change in rectangle parameters and passed into the state. 

What is a normalized action value? Say we have set n_actions = 101, then action values will be normalized by taking the actual action value and subtracting 51 from it so that we have a range of integer values from -50 to 50. The normalized action value will be added to the dimensions or coordinates of the rectangle.

With this approach, we don't let the agent produce anything other than disjoint rectangle sets. Thus, our reward only depends on number of rectangles killed.

**Change Required in Generate Session function** :

For the generate_sessions function, we need to give the initial state. For the initial state to be given, we need to formulate a method that evenly spaces N rectangles in the bounded region, and convert this to our own state representation which includes the interval graphs for x and y. Making this method needs to be our next step, after which we can simply call it and populate the first column of the state matrix (take care of how you do this because the format is not inherently clear) 

**The state representation is as follows** :

The observation space, or state dimensions are 

( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + ( N * 5 )

Let's break this down

1. region_bound * N * 4 - Contains the rectangle set representation, same as the one-hot method we have used before which follows directly from Wagner's template : (x1, x2, y1, y2)
2. 2 * N * N - Interval Graphs of X and Y intervals
3. 2 * N - The first N bits represent which rectangles have already been worked on (1) and which ones haven't (0) - This is primarily for the benefit of every 5th decision where the agent chooses the next rectangle to work on. The next N bits represent which rectangle is currently being worked on (1) - one hot encoding
4. N * 5 - which decision are we currently working on - one hot encoding

**Observations from the runs** :

Run 1 - 12 Rectangles, Adding/Subtracting width or height : Too many rectangles with extremely small dimensions - gets stuck at 1 killed

Run 2 - 12 Rectangles, Learning rate = 0.0001, Percentile 85, Reassigning width or height - In 9 iterations, the NN has gone from producing most generations with 1 rectangle killed to most with 2 rectangles killed and ONE with 3 rectangles killed. However, there have only been two generations with 3 killed rectangles until the 150th iteration

Run 3 - 12 Rectangles, Learning rate = 0.0001, Percentile 93, Super percentile 95, Reassigning width or height, reward scaling 100 - 35 iterations for first generation with 3 kills

Run 4 - 16 Rectangles, Learning rate = 0.0001, Percentile 93, Super percentile 95, Reassigning width or height, reward scaling 100 - 0 iterations for first generation with 3 kills. By the 30th iteration, almost all the elite rewards have 3 rectangles killed. We did not move past this due to a power cut

Run 5 - Simply a re-run of Run 4

Run 6 - 10 Rectangles, same parameters as run 4
Run 7 - 11 Rectangles, same parameters as run 4

Run 8 - 12 Rectangles, same
Run 9 - 13 Rectangles, same

Run 10 - 14 Rectangles, same
Run 11 - 15 Rectangles, same

# running
Run 12 - 17 Rectangles, same
Run 13 - 18 Rectangles, same

# to do
Run 14 - 19 Rectangles, same - cut short by power cut
Run 15 - 19 Rectangles, same - plateaus after 100 or so sessions after which it isn't able to produce more examples of 4 kill sets 

Run 16 - 20 Rectangles, same - only produced two constructions of max kills 5 till the 100th session. Showing no more examples of kills = 5 after the 30th session. In fact, even the number of 4 kill sets is quite low. This may improve with a lower learning rate or perhaps more layers in the network?

Run 17 - 20 Rectangels, same as 16 with 0.00005 learning rate instead of 0.0001

**Another approach - this is currently included** to solving the problem in larger numbers of rectangles could be to keep generating the first iteration again and again until at least one lucky hit is found. We only move on to training once we have a lucky hit.

*Hyperparameters used in run 1*

| Variable Name | Value | Significance |
|--|--|--|
| N | 7 | # Rectangles to generate |
| DECISIONS | N*4 | # 4 coordinates to be decided for each rectangle in generation |
 observation_space |  2*DECISIONS | # 4 coordinates to be decided for each rectangle in generation |
| LEARNING_RATE | 0.0005 | Learning Rate |
| n_sessions | 1000 | Batch Size / # Episodes or Generations per Sessions |
| Percentile | 90 | Top 100-X percent we are learning from after each session |
| super_percentile | 90 | Top 100-X percent of episodes that survive to the next iteration |
| FIRST_LAYER_NEURONS | 256 | - |
| SECOND_LAYER_NEURONS | 128 | - |
| THIRD_LAYER_NEURONS  | 128 | - |
| n_actions | 100 | The action space consists of all integers in [0,100). These will be the rectangle coordinates |
| disjoint_penalty | -2000 | Penalty for not generating a disjoint set of rectangles |  
| reward_scaling | 40 | Scaling factor. We multiply no of rectangles killed by the optimal cut sequence |
| long_episode_penalty | -10000 | Penalty for letting an episode run beyond DECISIONS*100 steps |
