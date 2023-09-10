# This is based on the code template for "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner

# This code is stated to work on tensorflow version 1.14.0 and python version 3.6.3
# for later versions, massive overheads are observed in the predict function for some reason, and/or it produces mysterious errors.

# Keras version 2.3.1 is used, not sure if this is important, but it comes recommended with the original code.

import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.losses import CategoricalCrossentropy
from statistics import mean
import pickle
import time
import math
import matplotlib.pyplot as plt

from OptimalCuts import Rectangle, intervalsIntersect, optimalCuts
from plotGenerations import plot_rectangles

N = 9  # Number of rectangles to be generated
DECISIONS = N*4  # For each rectangle, we generate 4 numbers within the bounded square region - the coordinates of the bottom-left and top-right corners  

LEARNING_RATE = 0.0005 # Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 2000 # number of new sessions per iteration - batch size
percentile = 70 # top 100-X percentile we are learning from
super_percentile = 90 # top 100-X percentile that survives to next iteration

# The original work revolves around generating a binary or n-ary sentence encoding
# However, here our action space is a continuous, finite interval from which any float number could be picked
# Therefore, we will try to take a different approach and see if that works

# Hidden Layer Neurons
FIRST_LAYER_NEURONS = 256
SECOND_LAYER_NEURONS = 128
THIRD_LAYER_NEURONS = 128
#FOURTH_LAYER_NEURONS = 128

reward_scaling = 40		# Scale reward to further incentivize killed rectangles

# At each step, the agent must pick an action which is an integer between 0 and 200
n_actions = 101		# integer values -50 to 50
region_bound = 200  # we will generate rectangles within a region enclosed by x = 0, x = 200, y = 0, y = 200

# Note for later:
# In case this is not exhaustive enough, we can either increase actions or divide this into 100 intervals of 
# length 1 each: 0-1, 1-2, ... 99-100 - The neural net will produce three numbers corresponding to each interval 
# # (P, M, V) - where P is the probability of picking a particular interval
# Within each interval, the number picked will be sampled from a Gaussian Distribution with mean = M and variance = V

observation_space = ( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + ( N * 5 )  # The state representation will have N*4 decisions, 2 N*N interval graph 
# representation and a one hot encoding of the current decision

len_game = DECISIONS 
state_dim = (observation_space,)

INF = 1000000

# We start by using ReLU for the Hidden Layers and Linear Activation for the output layer
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
#model.add(Dense(FOURTH_LAYER_NEURONS, activation="relu"))
model.add(Dense(n_actions, activation="softmax"))  # Alternatively, we can add softmax to this later just to maintain numerical stability during training
model.build((None, observation_space))
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate = LEARNING_RATE)) #Adam optimizer also works well, with lower learning rate

print(model.summary())

# Changes for N actions instead of 2 need to be incorporated 

def rectsFromState(state):
	# returns zero-area rectangle boolean, Rectangles
	rects = []
	i = 0

	while (i < DECISIONS):

		rect = [-1, -1, -1, -1]

		for k in range(region_bound):
			#look for x1, x2, y1, y2
			if state[i*(region_bound) + k] == 1 : rect[0] = k
			if state[i*(region_bound) + region_bound + k] == 1 : rect[1] = k
			if state[i*(region_bound) + (2*region_bound) + k] == 1 : rect[2] = k
			if state[i*(region_bound) + (3*region_bound) + k] == 1 : rect[3] = k
		
		#the rectangle must not be a zero area rectangle
		if (rect[0] == rect[1] or rect[2] == rect[3]):
			return False, rects
		
		rects.append(Rectangle( bottomLeft = ( rect[0], rect[1] ), topRight = ( rect[2], rect[3] ) ))
		i += 4

	return True, rects

def calc_score(state):
	
	# Reward function
	rectsFromStateResult = rectsFromState(state)

	# If some rectangles are zero area rectangles, return large negative reward
	if (not rectsFromStateResult[0]):
		return -20000

	rectangles = rectsFromStateResult[1]
	
	# Apply optimal cuts algorithm
	optimalCutsResult = optimalCuts(rectangles, Rectangle( bottomLeft = (0,0), topRight = (region_bound, region_bound)))
	
	# If generated rectangles are not disjoint, return big negative reward
	if (not optimalCutsResult[0]):
		return -1000

	# For disjoint sets, reward is proportional to number of killed rectangles
	return optimalCutsResult[2] * reward_scaling


def play_game(n_sessions, actions, state_next, states, prob, step, total_score):
	# plays one step concurrently for each of the n active sessions being made by generate_sessions
	for i in range(n_sessions):
		
		# generate integer action from [0, n_actions) based on probability scores
		action = np.random.choice(a = n_actions, p = prob[i])
		state_next[i] = states[i,:,step-1]	# get current state representation

		'''
		actions[i][step-1] = action
		state_next[i][ (n_actions*(step-1)) + action ] = 1	# supply state with current action taken
		state_next[i][ (n_actions*DECISIONS) + (2*N*N) + step-1 ] = 0 # current action already taken
		'''

		if (step % 5 == 0):		# choose the rectangle
			
			choice = action % N
			
			# choose the closest unworked rectangle
			diff = 100000
			for rect in range(N):
				if (state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + rect] == 0 and abs(rect - choice) < abs(diff)):
					diff = rect - choice

			choice += diff	# readjust to closest available rectangle
			
			actions[i][step-1] = choice		# record chosen action
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + choice] = 1	# Mark current rectangle as taken
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + N + choice] = 1	# Mark current rectangle as the one in use
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step-1] = 0 	# current decision already taken
	
		elif (step % 5 == 1):	# move bottom right corner x coordinate

			#determine which rectangle agent is working on
			rect_choice = 0

			while (rect_choice < N):
				if ( state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + N + rect_choice] == 1 ):
					break
				rect_choice += 1

			rectangles = rectsFromState(state_next[i])
			normalized_action = action - (n_actions//2)
			normalized_action_upper_bound = region_bound - 1 - rectangles[rect_choice].topRight[0]
			normalized_action_lower_bound = 0 - rectangles[rect_choice].bottomLeft[0]

			# check for
			for rect in range(N):
				
				if (rect == rect_choice):	# don't compare with self
					continue

				# check if the y intervals intersect - directly available in state representation
				if (state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] == 1):
					
					if (rectangles[rect_choice].topRight[0] <= rectangles[rect].bottomLeft[0]):  # possible collision for positive normalized actions  
						normalized_action_upper_bound = min( normalized_action_upper_bound,  rectangles[rect].bottomLeft[0] - rectangles[rect_choice].topRight[0] )

					if (rectangles[rect_choice].bottomLeft[0] >= rectangles[rect].topRight[0]):	# possible collision for negative normalized actions
						normalized_action_lower_bound = max( normalized_action_lower_bound,  rectangles[rect].topRight[0] - rectangles[rect_choice].bottomLeft[0] )


			normalized_action = max(normalized_action_lower_bound, normalized_action)
			normalized_action = min(normalized_action_upper_bound, normalized_action)
			true_action = normalized_action + (n_actions//2)
			x1 = rectangles[rect_choice].bottomLeft[0] + normalized_action
			x2 = rectangles[rect_choice].topRight[0] + normalized_action

			# fill in the x interval graph as it may be changed now
			for rect in range(N):
				
				if (rect == rect_choice):	# will always intersect with itself, no change required here
					continue

				if (intervalsIntersect((x1, x2), (rectangles[rect].bottomLeft[0], rectangles[rect].topRight[0]))):
					state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] = 1
					state_next[i][(region_bound * N *4) + (rect*N) + rect_choice] = 1

				else:
					state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] = 0
					state_next[i][(region_bound * N *4) + (rect*N) + rect_choice] = 0

			# Update state representation
			for k in range(region_bound):
				# look for x1, x2
				if (k == x1):
					state_next[i][rect_choice*(region_bound) + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + k] = 0
				
				if (k == x2):
					state_next[i][rect_choice*(region_bound) + region_bound + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + region_bound + k] = 0

			# update arrays
			actions[i][step-1] = true_action		# record chosen action
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step-1] = 0 	# current decision already taken

		elif (step % 5 == 2):	# move bottom right corner y coordinate

			#determine which rectangle agent is working on
			rect_choice = 0

			while (rect_choice < N):
				if ( state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + N + rect_choice] == 1 ):
					break
				rect_choice += 1

			rectangles = rectsFromState(state_next[i])
			normalized_action = action - (n_actions//2)
			normalized_action_upper_bound = region_bound - 1 - rectangles[rect_choice].topRight[1]
			normalized_action_lower_bound = 0 - rectangles[rect_choice].bottomLeft[1]

			# check for
			for rect in range(N):
				
				if (rect == rect_choice):	# don't compare with self
					continue

				# check if the x intervals intersect - directly available in state representation
				if (state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] == 1):
					
					if (rectangles[rect_choice].topRight[1] <= rectangles[rect].bottomLeft[1]):  # possible collision for positive normalized actions  
						normalized_action_upper_bound = min( normalized_action_upper_bound,  rectangles[rect].bottomLeft[1] - rectangles[rect_choice].topRight[1] )

					if (rectangles[rect_choice].bottomLeft[1] >= rectangles[rect].topRight[1]):	# possible collision for negative normalized actions
						normalized_action_lower_bound = max( normalized_action_lower_bound,  rectangles[rect].topRight[1] - rectangles[rect_choice].bottomLeft[1] )


			normalized_action = max(normalized_action_lower_bound, normalized_action)
			normalized_action = min(normalized_action_upper_bound, normalized_action)
			true_action = normalized_action + (n_actions//2)
			y1 = rectangles[rect_choice].bottomLeft[1] + normalized_action
			y2 = rectangles[rect_choice].topRight[1] + normalized_action

			# fill in the x interval graph as it may be changed now
			for rect in range(N):
				
				if (rect == rect_choice):	# will always intersect with itself, no change required here
					continue

				if (intervalsIntersect((y1, y2), (rectangles[rect].bottomLeft[1], rectangles[rect].topRight[1]))):
					state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] = 1
					state_next[i][(region_bound * N *4) + (N*N) + (rect*N) + rect_choice] = 1

				else:
					state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] = 0
					state_next[i][(region_bound * N *4) + (N*N) + (rect*N) + rect_choice] = 0

			# Update state representation
			for k in range(region_bound):
				# look for y1, y2
				if (k == y1):
					state_next[i][rect_choice*(region_bound) + (2*region_bound) + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + (2*region_bound) + k] = 0
				
				if (k == y2):
					state_next[i][rect_choice*(region_bound) + (3*region_bound) + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + (3*region_bound) + k] = 0

			# update arrays
			actions[i][step-1] = true_action		# record chosen action
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step-1] = 0 	# current decision already taken

		elif (step % 5 == 3):	# x scaling (width scaling)

			#determine which rectangle agent is working on
			rect_choice = 0

			while (rect_choice < N):
				if ( state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + N + rect_choice] == 1 ):
					break
				rect_choice += 1

			rectangles = rectsFromState(state_next[i])
			normalized_action = action - (n_actions//2)
			normalized_action_upper_bound = region_bound - 1 - rectangles[rect_choice].topRight[0]
			normalized_action_lower_bound = 1 - rectangles[rect_choice].bottomLeft[0]

			# check for
			for rect in range(N):
				
				if (rect == rect_choice):	# don't compare with self
					continue

				# check if the y intervals intersect - directly available in state representation
				if (state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] == 1):
					
					if (rectangles[rect_choice].topRight[0] <= rectangles[rect].bottomLeft[0]):  # possible collision for positive normalized actions  
						normalized_action_upper_bound = min( normalized_action_upper_bound,  rectangles[rect].bottomLeft[0] - rectangles[rect_choice].topRight[0] )

			normalized_action = max(normalized_action_lower_bound, normalized_action)
			normalized_action = min(normalized_action_upper_bound, normalized_action)
			true_action = normalized_action + (n_actions//2)
			x1 = rectangles[rect_choice].bottomLeft[0]	# anchored to bottom left, this co-ordinate will not change
			x2 = rectangles[rect_choice].topRight[0] + normalized_action

			# fill in the x interval graph as it may be changed now (this is true even if width decreases)
			for rect in range(N):
				
				if (rect == rect_choice):	# will always intersect with itself, no change required here
					continue

				if (intervalsIntersect((x1, x2), (rectangles[rect].bottomLeft[0], rectangles[rect].topRight[0]))):
					state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] = 1
					state_next[i][(region_bound * N *4) + (rect*N) + rect_choice] = 1

				else:
					state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] = 0
					state_next[i][(region_bound * N *4) + (rect*N) + rect_choice] = 0

			# Update state representation
			for k in range(region_bound):
				# no need to change x1 as it is anchored already
				if (k == x2):
					state_next[i][rect_choice*(region_bound) + region_bound + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + region_bound + k] = 0

			# update arrays
			actions[i][step-1] = true_action		# record chosen action
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step-1] = 0 	# current decision already taken


		else: # step % 5 == 4 ---> y scaling (height scaling)

			#determine which rectangle agent is working on
			rect_choice = 0

			while (rect_choice < N):
				if ( state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + N + rect_choice] == 1 ):
					break
				rect_choice += 1

			rectangles = rectsFromState(state_next[i])
			normalized_action = action - (n_actions//2)
			normalized_action_upper_bound = region_bound - 1 - rectangles[rect_choice].topRight[1]
			normalized_action_lower_bound = 1 - rectangles[rect_choice].bottomLeft[1]

			# check for
			for rect in range(N):
				
				if (rect == rect_choice):	# don't compare with self
					continue

				# check if the x intervals intersect - directly available in state representation
				if (state_next[i][(region_bound * N *4) + (rect_choice*N) + rect] == 1):
					
					if (rectangles[rect_choice].topRight[1] <= rectangles[rect].bottomLeft[1]):  # possible collision for positive normalized actions  
						normalized_action_upper_bound = min( normalized_action_upper_bound,  rectangles[rect].bottomLeft[1] - rectangles[rect_choice].topRight[1] )

			normalized_action = max(normalized_action_lower_bound, normalized_action)
			normalized_action = min(normalized_action_upper_bound, normalized_action)
			true_action = normalized_action + (n_actions//2)
			y1 = rectangles[rect_choice].bottomLeft[1]	# no changes, this is anchored
			y2 = rectangles[rect_choice].topRight[1] + normalized_action

			# fill in the y interval graph as it may be changed now - once again, this applies even if height is decreased
			for rect in range(N):
				
				if (rect == rect_choice):	# will always intersect with itself, no change required here
					continue

				if (intervalsIntersect((y1, y2), (rectangles[rect].bottomLeft[1], rectangles[rect].topRight[1]))):
					state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] = 1
					state_next[i][(region_bound * N *4) + (N*N) + (rect*N) + rect_choice] = 1

				else:
					state_next[i][(region_bound * N *4) + (N*N) + (rect_choice*N) + rect] = 0
					state_next[i][(region_bound * N *4) + (N*N) + (rect*N) + rect_choice] = 0

			# Update state representation
			for k in range(region_bound):
				# no need to change y1, this is anchored
				if (k == y2):
					state_next[i][rect_choice*(region_bound) + (3*region_bound) + k] = 1
				else:
					state_next[i][rect_choice*(region_bound) + (3*region_bound) + k] = 0

			# update arrays
			actions[i][step-1] = true_action		# record chosen action
			state_next[i][( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step-1] = 0 	# current decision already taken

		if (step < DECISIONS):	# doesn't make sense to make the next decision position 1 if already terminal	
			state_next[i][ ( region_bound * N * 4 ) + ( 2 * N * N ) + ( 2 * N ) + step ] = 1		
		
		#calculate final score
		terminal = step == DECISIONS
		if terminal:
			total_score[i] = calc_score(state_next[i])
	
		# record sessions 
		if not terminal:
			states[i,:,step] = state_next[i]
		
	return actions, state_next,states, total_score, terminal	
					

def generate_session(agent, n_sessions, verbose = 1):	
	"""
	Play n_session games using agent neural network.
	Terminate when games finish 
	"""
	# states - the initial state still needs to be given

	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int) # all states encountered in all sessions
	actions = np.zeros([n_sessions, len_game], dtype = int)	 # all actions taken in all sessions
	state_next = np.zeros([n_sessions,observation_space], dtype = int) # current state of each session
	prob = np.zeros(n_sessions) # action probabilities for current state of each session
	states[:,(n_actions*DECISIONS) + (2*N*N),0] = 1
	step = 0
	total_score = np.zeros([n_sessions])
	pred_time = 0
	play_time = 0
	
	while (True):
		step += 1		
		tic = time.time()
		prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
		pred_time += time.time()-tic
		tic = time.time()
		actions, state_next,states, total_score, terminal = play_game(n_sessions, actions,state_next,states,prob, step, total_score)
		play_time += time.time()-tic
		
		if terminal:
			break
	if (verbose):
		print("Predict: "+str(pred_time)+", play: " + str(play_time))
	#last returned variable is meant to return the final generation
	return states, actions, total_score, state_next 
	

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
	"""
	Select states and actions from games that have rewards >= percentile
	:param states_batch: list of lists of states, states_batch[session_i][t]
	:param actions_batch: list of lists of actions, actions_batch[session_i][t]
	:param rewards_batch: list of rewards, rewards_batch[session_i]

	:returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions
	
	This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	elite_states = []
	elite_actions = []
	elite_rewards = []
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:		
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				for item in states_batch[i]:
					elite_states.append(item.tolist())
				for item in actions_batch[i]:
					elite_actions.append(item)			
			counter -= 1
	elite_states = np.array(elite_states, dtype = int)	
	elite_actions = np.array(elite_actions, dtype = int)	
	return elite_states, elite_actions
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, generations_batch, percentile=90):
	"""
	Select all the sessions that will survive to the next generation
	Similar to select_elites function
	If this function is the bottleneck, it can easily be sped up using numba
	"""
	
	counter = n_sessions * (100.0 - percentile) / 100.0
	reward_threshold = np.percentile(rewards_batch,percentile)

	super_states = []
	super_actions = []
	super_rewards = []
	super_generations = []

	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				super_generations.append(generations_batch[i])
				counter -= 1

	super_states = np.array(super_states, dtype = int)
	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	super_generations = np.array(super_generations)

	return super_states, super_actions, super_rewards, super_generations
	

super_states =  np.empty((0,len_game,observation_space), dtype = int)
super_generations = np.array([], dtype=int)
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0

myRand = 3 # run number used in the filename

'''
sessions = generate_session(model,1,0)	# Play one episode and evaluate it

states_batch = np.array(sessions[0], dtype = int)
actions_batch = np.array(sessions[1], dtype = int)
rewards_batch = np.array(sessions[2])
generations_batch = np.array(sessions[3])
states_batch = np.transpose(states_batch,axes=[0,2,1])

print(actions_batch)
print(sessions[3].shape)

print(" Interval Graph Representation : ")
x_interval_graph = " ".join([str(sessions[3][0][ (n_actions * DECISIONS) + i ]) for i in range(N*N)])
y_interval_graph = " ".join([str(sessions[3][0][ (N*N) + (n_actions * DECISIONS) + i ]) for i in range(N*N)])

print(f"X : {x_interval_graph}")
print(f"Y : {y_interval_graph}")

rectsGenerated = rectsFromState(sessions[3][0])
print(rectsGenerated)

print()

if (rectsGenerated[0]):
	# Not zero-area
	result = optimalCuts(rectsGenerated[1], Rectangle( bottomLeft = (0,0), topRight = (region_bound, region_bound)))
	print(calc_score(sessions[3][0]))
	plot_rectangles(rectsGenerated[1], result[2], myRand, 0, region_bound)
	print("On applying the cutting algorithm we find")
	print(result)

super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, generations_batch, percentile=super_percentile) #pick the sessions to survive
print()

print("Super generations array before is :")
print(super_sessions[3])

super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i], super_sessions[3][i]) for i in range(len(super_sessions[2]))]
super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
super_generations = [super_sessions[i][3] for i in range(len(super_sessions))]

print(super_generations[-1].shape)

#####################################

for i in range(1000000): #1000000 generations should be plenty
	#generate new sessions
	#performance can be improved with joblib
	tic = time.time()
	sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
	
	rewards_batch = np.array(sessions[2])
	repeat_gens = 1
	
	# make sure there is at least one disjoint set at the start
	while (i == 0 and max(rewards_batch) < 0):
		repeat_gens += 1
		sessions = generate_session(model,n_sessions,0)
		rewards_batch = np.array(sessions[2])

	sessgen_time = time.time()-tic
	tic = time.time()
	
	if (i == 0):
		print(f"First iteration required {sessgen_time} and {repeat_gens} session generations")

	states_batch = np.array(sessions[0], dtype = int)
	actions_batch = np.array(sessions[1], dtype = int)
	generations_batch = np.array(sessions[3])
	states_batch = np.transpose(states_batch,axes=[0,2,1])
	states_batch = np.append(states_batch,super_states,axis=0)

	#print(f"Iteration: {i}")
	#print(f"States batch shape = {states_batch.shape}")

	if i>0:
		actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
	rewards_batch = np.append(rewards_batch,super_rewards)
	
	#print(f"Super generations shape {super_generations.shape}")
	#print(f"Generations batch shape = {generations_batch.shape}")

	if (i > 0):
		generations_batch = np.append(generations_batch, super_generations, axis=0)
		#print(f"Generations batch shape after append = {generations_batch.shape}")
		
	randomcomp_time = time.time()-tic 
	tic = time.time()

	# elite_actions is a numpy array
	elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
	select1_time = time.time()-tic

	tic = time.time()
	super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, generations_batch, percentile=super_percentile) #pick the sessions to survive
	select2_time = time.time()-tic
	
	tic = time.time()
	super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i], super_sessions[3][i]) for i in range(len(super_sessions[2]))]
	super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
	select3_time = time.time()-tic

	#one hot encoding for elite actions
	elite_actions_one_hot = np.zeros((elite_actions.size, n_actions))
	elite_actions_one_hot[np.arange(elite_actions.size), elite_actions] = 1
	
	tic = time.time()
	model.fit(elite_states, elite_actions_one_hot) #learn from the elite sessions
	fit_time = time.time()-tic
	
	tic = time.time()
	
	super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
	super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
	super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
	super_generations = np.array([super_sessions[i][3] for i in range(len(super_sessions))])
	
	rewards_batch.sort()
	mean_all_reward = np.mean(rewards_batch[-100:])	
	mean_best_reward = np.mean(super_rewards)	

	score_time = time.time()-tic
	
	print("\n" + str(i) +  ". Best individuals: " + str(np.flip(np.sort(super_rewards))))
	
	#uncomment below line to print out how much time each step in this loop takes. 
	print(	"Mean reward: " + str(mean_all_reward) + "\nSessgen: " + str(sessgen_time) + ", other: " + str(randomcomp_time) + ", select1: " + str(select1_time) + ", select2: " + str(select2_time) + ", select3: " + str(select3_time) +  ", fit: " + str(fit_time) + ", score: " + str(score_time)) 
	
	if (i%20 == 0): #Write all important info to files every 20 iterations
		with open('run_info/best_species_pickle_'+str(myRand)+'.txt', 'wb') as fp:
			pickle.dump(super_actions, fp)
		with open('run_info/best_species_txt_'+str(myRand)+'.txt', 'w') as f:
			for item in super_actions:
				f.write(str(item))
				f.write("\n")
		with open('run_info/best_species_rewards_'+str(myRand)+'.txt', 'w') as f:
			for item in super_rewards:
				f.write(str(item))
				f.write("\n")
		with open('run_info/best_100_rewards_'+str(myRand)+'.txt', 'a') as f:
			f.write(str(mean_all_reward)+"\n")
		with open('run_info/best_elite_rewards_'+str(myRand)+'.txt', 'a') as f:
			f.write(str(mean_best_reward)+"\n")

	if (i%200==2): # To create a timeline, like in Figure 3
		with open('best_species_timeline_txt_'+str(myRand)+'.txt', 'a') as f:
			f.write(str(super_actions[0]))
			f.write("\n")
			
	if (i%50 == 0):	# Make a plot of best generation every 50th iteration
		rectangles = rectsFromState(super_generations[0])
		plot_rectangles(rectangles[1], super_rewards[0]/reward_scaling, i, 0, region_bound, myrand = myRand)		# Scale reward to further incentivize killed rectangles
'''