# Run Observations 

**The Current rectangle generation paradigm is as follows** :

We allow the agent to take N*4 decisions where N = number of rectangles we want to generate. For each rectangle, the agent produces 4 co-ordinates in succession - (x, y, w, h) where (x, y) is the bottom-left corner of the rectangle and (w, h) is the width and height respectively. 

While actions are chosen in [0, 99], we make sure that actions chosen for (w, h) are corrected by +1 - this way there is no chance of zero area rectangles being generated at all.

This way, the only thing that needs to be changed is the rectangle generation function.

For non disjoint rectangle sets we give a high negative reward. For disjoint sets, reward is the number of rectangles killed by the optimal cut algorithm scaled by a scaling factor.

**The state representation is as follows** :

The observation space, or state dimensions are ((n_actions + 1) * DECISIONS). Following from Wagner's code template, the first *n_actions * DECISIONS* state parameters are one-hot representations of the actual decisions taken. i.e. the rectangle co-ordinates and dimensions generated. While the next *DECISIONS* state parameters are a one-hot-encoding of which decision is currently being made.

**Observations from the run** :

run1 = 7 rectangles, reward_scale 10

run 2 = 9 rectangles, reward_scale 40 - We know there is at least one generation of 9 rectangles that has 2 killed rectangles with the optimal cut sequence, so we can observe if this gets picked up on

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
