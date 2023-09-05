# Run Observations

**The Current rectangle generation paradigm is as follows** :

We allow the agent to take N*4 decisions where N = number of rectangles we want to generate. For each rectangle, the agent produces 4 co-ordinates in succession - (x1, x2, y1, y2) where (x1, y1) is the bottom-left corner of the rectangle and (x2, y2) is the top-right corner.

For zero-area rectangles generated, we give a high negative reward. For non disjoint rectangle sets this reward is slightly less negative. For disjoint sets, reward is the number of rectangles killed by the optimal cut algorithm scaled by a scaling factor.

**The state representation is as follows** :

The observation space, or state dimensions are ((n_actions + 1) * DECISIONS). Following from Wagner's code template, the first *n_actions * DECISIONS* state parameters are one-hot representations of the actual decisions taken. i.e. the rectangle co-ordinates generated. While the next *DECISIONS* state parameters are a one-hot-encoding of which decision is currently being made.

**Observations from the runs** :

Run 1 - 7 rectangles , action space 100, 1000 episodes per iteration - In the first iteration there is ONE lucky shot. The agent seems to pick up on disjoint sets quickly but it takes time to generate sets with at least 1 killed rectangle. We have not progressded from 1 to 2 killed rectangles till now (~4600 iterations). There is an apparent need to scale the reward, so that kills are incentivized more heavily. The difference between 0 and -1000 is far greater than 1 and 0 - we will try scaling by 10 

Run 2 - 12 rectangles, action space 200, reward scaling 10, 1000 episodes per iteration - this action space still feels too small for a disjoint set to randomly occur in one of the first few runs - could we put an upper bound on rectangle areas? - Perhaps the sum of rectangle areas? - or also pass the area into the state representation as decisions are made?

Run 3 - 12 rectangles, action space 400, reward scaling 10, 2000 episodes per iteration, percentile 60, super_percentile 90 - it seems that scaling up the action space doesn't really help the agent get a lucky shot at a disjoint set in the early sessions. If we look at the three generations rects_1, rects_2, rects_3 - all of them look equally cluttered. A **possible approach** of rectangle reresentation could be - Instead of generating (x1, x2, y1, y2) we will generate (x1, y1, width, height) - perhaps it will be easier for the agent to generate a disjoint set when it generates rectangles as objects anchored to a single point with two dimensions extending from that point.

Run 4 -  7 rectangles , action space 100, reward scaling 10, percentile 70, super_percentile 90, 2000 episodes per iteration - In the first iteration there is ONE lucky shot at a disjoint set. We notice that the frequency of generations with at least one rectangle killed increases at a greater rate than it did for run 1 where there was no reward scaling. However, this increase in frequency is still on the slower side. Perhaps we will also see generations with two or more rectangles being killed eventually. 

We have produced an image of the best rectangle generation for every 50th session. These are stored in plots/run_4

Run 5 - 9 rectangles, reward scaling 40 - extremely slow. Similar to the case in interval graph representation branch. We need to either increase the learning rate or increase the number of layers in the neural network to see what exactly the issue is

**Another approach** to solving the problem in larger numbers of rectangles could be to keep generating the first iteration again and again until at least one lucky hit is found. We only move on to training once we have a lucky hit.

Hyperparameters used in the first run

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
