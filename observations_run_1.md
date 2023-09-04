# Run Observations

**The Current rectangle generation paradigm is as follows** :

We allow the agent to take N*4 decisions where N = number of rectangles we want to generate. For each rectangle, the agent produces 4 co-ordinates in succession - (x1, x2, y1, y2) where (x1, y1) is the bottom-left corner of the rectangle and (x2, y2) is the top-right corner.

For zero-area rectangles generated, we give a high negative reward. For non disjoint rectangle sets this reward is slightly less negative. For disjoint sets, reward is the number of rectangles killed by the optimal cut algorithm scaled by a scaling factor.

**The state representation is as follows** :

The observation space, or state dimensions are (n_actions * DECISIONS) + (2 * N * N) + DECISIONS. Following from Wagner's code template, the first *n_actions * DECISIONS* state parameters are one-hot representations of the actual decisions taken. i.e. the rectangle co-ordinates generated. The next N * N parameters will be a binary representation of the x interval graph. Similarly, the next N * N will contain the y interval graph. The last *DECISIONS* state parameters are a one-hot-encoding of which decision is currently being made.

**Observations from the runs** :

No runs yet. This will be updated

**Another approach** to solving the problem in larger numbers of rectangles could be to keep generating the first iteration again and again until at least one lucky hit is found. We only move on to training once we have a lucky hit.

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
