# RL-Disprove-Mathematical-Conjectures
This repository contains all the code for a research project on Disproving Mathematical Conjectures using Reinforcement Learning

We observe that the Neural Network, in its untrained state, is prone to generating the same action values consecutively. This isn't very surprising as the state representation given to it is also similar from one step to the next. Only two numbers in the state representation change from one step to the next. 

This is a problem because generating the same action values in close proximity can lead to x1 == x2 or y1 == y2 where (x1, y1) and (x2, y2) are the diagonally opposite rectangle corners of the rectangle we are generating. We see that in all our runs, exactly this ends up being the case for every generation produced in the first batch of episodes (NN entirely untrained). We tried to mitigate this by assigning a very high negative reward to episodes where this occurs. However, the occurence itself is disastrous especially for the cross entropy approach since the training depends on taking the top x percentile of episodes from the current batch and shifting the policy towards these. If all episodes have the same large negative reward to begin with, then there's no point learning from the top x% of them.  

To address this issue, we will abandon the current approach of generating all episodes in a batch concurrently and allow episodes to be of variable length. This will create a time overhead, so why are we doing this?

In case the agent decides to take a duplicate consecutive action for a (x1, x2) or (y1, y2) pair, we can let the state remain unchanged for the next step. i.e. the rectangle generation halts until a distinct integer gets picked. To discourage extremely long episodes from becoming pervasive, we will subtract a delay factor from the reward at the end of each episode - which will be proportional to the number of extra steps taken. 

We are doing this in a separate branch "variable-length-episodes" as it requires a significant amount of refactoring