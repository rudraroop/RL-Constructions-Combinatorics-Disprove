# This will be updated after we actually manage to get a successful run

Run 1 - 7 rectangles , action space 100 - the agent seems to pick up on disjoint sets quickly but it takes time to generate sets with at least 1 killed rectangle. We have not progressded from 1 to 2 killed rectangles till now (1000 iterations). There is an apparent need to scale the reward, so that kills are incentivized more heavily. The difference between 0 and -1000 is far greater than 1 and 0 - we will try scaling by 10 

Run 2 - 12 rectangles, action space 200 - this action space still feels too small for a disjoint set to randomly occur in one of the first few runs - could we put an upper bound on rectangle areas? - Perhaps the sum of rectangle areas? - or also pass the area into the state representation as decisions are made?

Run 3 - 12 rectangles, action space 400 - it seems that scaling up the action space doesn't really help the agent get a lucky shot at a disjoint set in the early sessions. If we look at the three generations rects_1, rects_2, rects_3 - all of them look equally cluttered. We will explore another way of rectangle reresentation in a separate feature branch. Instead of generating (x1, x2, y1, y2) we will generate (x1, y1, width, height) - perhaps it will be easier for the agent to generate a disjoint set when it generates rectangles as objects anchored to a single point with two dimensions extending from that point.

**The Current rectangle generation paradigm is as follows** :

We allow the agent to take N*4 decisions where N = number of rectangles we want to generate. For each rectangle, the agent produces 4 co-ordinates in succession - (x1, x2, y1, y2) where (x1, y1) is the bottom-left corner of the rectangle and (x2, y2) is the top-right corner.

The episodes are variable-length. We want to make sure that the agent doesn't take two consecutive decisions where x1 == x2 or y1 == y2 as we have observed this tendency before and this makes the produced rectangle a zero-area rectangle.  Thus we don't count such actions as valid decisions. The game doesn't progress if such an action is taken. In the reward function, we subtract a **delay factor** (proportional to the episode length) from the reward to discourage behaviour where zero-area rectangles are being produced continuously, not allowing the episode to progress.

**The state representation is as follows** :

The observation space, or state dimensions are (DECISIONS*2). Following from Wagner's code template, the first *DECISIONS* state parameters are the actual decisions taken. i.e. the rectangle co-ordinates generated. While the next *DECISIONS* state parameters are a one-hot-encoding of which decision is currently being made.

**Observations from the run** :



Hyperparameters used in this run

| Variable Name | Value | Significance |
|--|--|--|
| N | 14 | # Rectangles to generate |
| DECISIONS | N*4 | # 4 coordinates to be decided for each rectangle in generation |
 observation_space |  2*DECISIONS | # 4 coordinates to be decided for each rectangle in generation |
| LEARNING_RATE | 0.0001 | Learning Rate |
| n_sessions | 400 | Batch Size / # Episodes or Generations per Sessions |
| Percentile | 70 | Top 100-X percent we are learning from after each session |
| super_percentile | 90 | Top 100-X percent of episodes that survive to the next iteration |
| FIRST_LAYER_NEURONS | 256 | - |
| SECOND_LAYER_NEURONS | 128 | - |
| THIRD_LAYER_NEURONS  | 128 | - |
| n_actions | 100 | The action space consists of all integers in [0,100). These will be the rectangle coordinates |
| disjoint_penalty | -2000 | Penalty for not generating a disjoint set of rectangles |  
| long_episode_penalty | -10000 | Penalty for letting an episode run beyond DECISIONS*100 steps |
