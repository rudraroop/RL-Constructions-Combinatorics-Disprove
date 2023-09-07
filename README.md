# RL-Disprove-Mathematical-Conjectures
This repository contains all the code for a research project on Disproving Mathematical Conjectures using Reinforcement Learning

In our previous approaches, such as the one-hot state representation approach and its offshoots - anchored rectangles and interval graph representation, we observed that assigning a highly negative reward to non-disjoint rectangle sets is extremely time consuming. The Neural Network tends to produce overlapping generations almost exclusively and we have to generate the first iteration multiple times to be able to produce a single disjoint set. 

Our next approach will be to constrain the generation itself, so that the NN only produces disjoint sets of rectangles. We will give the NN a state representation of a disjoint N rectangle set to begin with. For each rectangle, the agent will choose to move its anchor point (bottom left) and scale its dimensions. If the scaling collides with a region boundary or the boundary of another rectangle, we will stop the scaling at this boundary.

This will happen as a sequence of 4 decisions taken per rectangle. We would ideally like to give the NN the ability to pick any of the rectangles at a given step, instead of having to proceed sequentially in the same order each time.