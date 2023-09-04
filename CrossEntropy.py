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

from OptimalCuts import Rectangle, optimalCuts
from plotGenerations import plot_rectangles

N = 7  # Number of rectangles to be generated
DECISIONS = N*4  # For each rectangle, we generate 4 numbers within the bounded square region - the coordinates of the bottom-left and top-right corners  

LEARNING_RATE = 0.0001 # Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
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

reward_scaling = 10		# Scale reward to further incentivize killed rectangles

# At each step, the agent must pick an action which is an integer between 0 and 200
n_actions = 100
region_bound = n_actions  # we will generate rectangles within a region enclosed by x = 0, x = 200, y = 0, y = 200

# Note for later:
# In case this is not exhaustive enough, we can either increase actions or divide this into 100 intervals of 
# length 1 each: 0-1, 1-2, ... 99-100 - The neural net will produce three numbers corresponding to each interval 
# # (P, M, V) - where P is the probability of picking a particular interval
# Within each interval, the number picked will be sampled from a Gaussian Distribution with mean = M and variance = V

observation_space = (n_actions + 1) * DECISIONS    # The state representation will have N*4 decisions, each decision
# is an action that will be one-hot-encoded within n actions. 

len_game = DECISIONS 
state_dim = (observation_space,)

INF = 1000000

# We start by using ReLU for the Hidden Layers and Linear Activation for the output layer
model = Sequential()
model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
model.add(Dense(n_actions, activation="softmax"))  # Alternatively, we can add softmax to this later just to maintain numerical stability during training
model.build((None, observation_space))
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate = LEARNING_RATE)) #Adam optimizer also works well, with lower learning rate

print(model.summary())

# Changes for N actions instead of 2 need to be incorporated 

def rectsFromState(state):
	rects = []
	i = 0
	
	while (i < DECISIONS):

		rect = [-1, -1, -1, -1]

		for k in range(n_actions):
			#look for x1
			if state[i*(n_actions) + k] == 1 : rect[0] = k
			if state[i*(n_actions) + n_actions + k] == 1 : rect[1] = k
			if state[i*(n_actions) + (2*n_actions) + k] == 1 : rect[2] = k
			if state[i*(n_actions) + (3*n_actions) + k] == 1 : rect[3] = k
		
		#the rectangle must not be a zero area rectangle
		if (rect[0] == rect[1] or rect[2] == rect[3]):
			return False, rects
		
		rects.append(Rectangle( bottomLeft = ( min(rect[0], rect[1]), min(rect[2], rect[3]) ), topRight = ( max(rect[0], rect[1]), max(rect[2], rect[3]) ) ))
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
		
		actions[i][step-1] = action
		state_next[i] = states[i,:,step-1]	# get current state representation
		state_next[i][ (n_actions*(step-1)) + action ] = 1	# supply state with current action taken
		state_next[i][ (n_actions*DECISIONS) + step-1 ] = 0 # current action already taken
		
		if (step < DECISIONS):	# doesn't make sense to make the next action position 1 if already terminal
			state_next[i][ (n_actions*DECISIONS) + step] = 1			
		
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
	
	Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
	"""
	states =  np.zeros([n_sessions, observation_space, len_game], dtype=int) # all states encountered in all sessions
	actions = np.zeros([n_sessions, len_game], dtype = int)	 # all actions taken in all sessions
	state_next = np.zeros([n_sessions,observation_space], dtype = int) # current state of each session
	prob = np.zeros(n_sessions) # action probabilities for current state of each session
	states[:,DECISIONS,0] = 1
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

myRand = 4 # run number used in the filename

'''
sessions = generate_session(model,100,0)	# Play one episode and evaluate it

states_batch = np.array(sessions[0], dtype = int)
actions_batch = np.array(sessions[1], dtype = int)
rewards_batch = np.array(sessions[2])
generations_batch = np.array(sessions[3])
states_batch = np.transpose(states_batch,axes=[0,2,1])

print(sessions[3])
print(actions_batch)

print()

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
'''

for i in range(1000000): #1000000 generations should be plenty
	#generate new sessions
	#performance can be improved with joblib
	tic = time.time()
	sessions = generate_session(model,n_sessions,0) #change 0 to 1 to print out how much time each step in generate_session takes 
	sessgen_time = time.time()-tic
	tic = time.time()
	
	states_batch = np.array(sessions[0], dtype = int)
	actions_batch = np.array(sessions[1], dtype = int)
	rewards_batch = np.array(sessions[2])
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
	