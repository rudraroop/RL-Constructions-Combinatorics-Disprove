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

N = 14  # Number of rectangles to be generated
DECISIONS = N*4  # For each rectangle, we generate 4 numbers within the bounded square region - the coordinates of the bottom-left and top-right corners  

LEARNING_RATE = 0.0001 # Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
n_sessions = 10000 # number of new sessions per iteration - batch size
percentile = 60 # top 100-X percentile we are learning from
super_percentile = 90 # top 100-X percentile that survives to next iteration
region_bound = 100  # we will generate rectangles within a region enclosed by x = 0, x = 100, y = 0, y = 100

# The original work revolves around generating a binary or n-ary sentence encoding
# However, here our action space is a continuous, finite interval from which any float number could be picked
# Therefore, we will try to take a different approach and see if that works

# Hidden Layer Neurons
FIRST_LAYER_NEURONS = 256
SECOND_LAYER_NEURONS = 128
THIRD_LAYER_NEURONS = 128

# At each step, the agent must pick an action which is an integer between 0 and 100
n_actions = 100
# Note for later:
# In case this is not exhaustive enough, we can either increase actions or divide this into 100 intervals of 
# length 1 each: 0-1, 1-2, ... 99-100 - The neural net will produce three numbers corresponding to each interval 
# # (P, M, V) - where P is the probability of picking a particular interval
# Within each interval, the number picked will be sampled from a Gaussian Distribution with mean = M and variance = V

observation_space = 2*DECISIONS #Leave this at 2*DECISIONS. The input vector will have size 2*DECISIONS, where the first DECISIONS letters encode our partial word (with zeros on
						  #the positions we haven't considered yet), and the next DECISIONS bits one-hot encode which letter we are considering now.
						  #So e.g. [10.0,12.3,0,0,   0,0,1,0] means we have the partial word 10.0,12.3 and we are considering the third letter now.
						  #If this doesn't work, we can supply interval graph data to the neural network as state information

#len_game = DECISIONS 
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
		
		#the rectangle must not be a zero area rectangle
		if (state[i] == state[i+1] or state[i+2] == state[i+3]):
			return False, rects
		
		rects.append(Rectangle( bottomLeft = ( min(state[i], state[i+1]), min(state[i+2], state[i+3]) ), topRight = ( max(state[i], state[i+1]), max(state[i+2], state[i+3]) ) ))
		i += 4

	return True, rects

def calc_score_variable_length(state, len_game, decision_idx):

	if (decision_idx != DECISIONS):
		# generation wasn't completed, return very large negative reward
		return -10000

	# Reward function
	rectsFromStateResult = rectsFromState(state)
	rectangles = rectsFromStateResult[1]
	
	delay_factor = len_game - DECISIONS

	# Apply optimal cuts algorithm
	optimalCutsResult = optimalCuts(rectangles, Rectangle( bottomLeft = (0,0), topRight = (100, 100)))
	
	# If generated rectangles are not disjoint, return big negative reward
	if (not optimalCutsResult[0]):
		return - 1000 - delay_factor 

	# For disjoint sets, reward is proportional to number of killed rectangles
	return (2*optimalCutsResult[2]) - (delay_factor/4)
	

def play_one_game(agent, begin_state, states, actions, episode_scores, final_generations):
	
	decision_idx = 0
	cur_state = begin_state
	prev_decision = -1
	len_game = 0

	game_states = []
	game_actions = []

	while (decision_idx < DECISIONS and len_game < DECISIONS*100):
		# make sure state has decision information
		cur_state[DECISIONS + decision_idx] = 1
		game_states.append(cur_state)
		# get probability distribution over actions based on current state
		prob = agent.predict(cur_state)
		# generate integer action from [0, 100) based on probability scores
		action = np.random.choice(a = n_actions, p = prob)
		game_actions.append(action)
		len_game += 1
		
		if (decision_idx % 4 == 1):
			# x2 is being generated, hence check if same as x1
			if (not prev_decision == action):
				# action is valid, doesn't cause a zero area rectangle to be generated - change state
				cur_state[decision_idx] = action
				cur_state[DECISIONS + decision_idx] = 0
				# record action as new prev_decision
				prev_decision = action
				# increment decision index
				decision_idx += 1
				
		elif (decision_idx % 4 == 3):
			# y2 is being generated, hence check if same as y1
			if (not prev_decision == action):
				# action is valid, doesn't cause a zero area rectangle to be generated - change state
				cur_state[decision_idx] = action
				cur_state[DECISIONS + decision_idx] = 0
				# record action as new prev_decision
				prev_decision = action
				# increment decision index
				decision_idx += 1

		else:
			# any action is valid, change state
			cur_state[decision_idx] = action
			cur_state[DECISIONS + decision_idx] = 0
			# record action as new prev_decision
			prev_decision = action
			# increment decision index
			decision_idx += 1	

	# episode terminated, compute reward
	score = calc_score_variable_length(cur_state, len_game, decision_idx)

	# add information to session arrays
	states.append(np.array(game_states))
	actions.append(np.array(game_actions))
	episode_scores.append(score)
	final_generations.append(cur_state)

	return game_states, game_actions, score, cur_state
	

def generate_session_sequential(agent, n_sessions, verbose = 1):
	
	states = np.array([])
	actions = np.array([])
	episode_scores = np.array([])
	final_generations = np.array([])

	for episode_idx in range(n_sessions):
		begin_state = np.zeros(state_dim, dtype = int)
		play_one_game(agent, begin_state, states, actions, episode_scores, final_generations)

	states = np.array(states)
	actions = np.array(actions)
	episode_scores = np.array(episode_scores)
	final_generations = np.array(final_generations)

	return states, actions, episode_scores, final_generations


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
	
def select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=90):
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
	for i in range(len(states_batch)):
		if rewards_batch[i] >= reward_threshold-0.0000001:
			if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
				super_states.append(states_batch[i])
				super_actions.append(actions_batch[i])
				super_rewards.append(rewards_batch[i])
				counter -= 1
	super_states = np.array(super_states, dtype = int)
	super_actions = np.array(super_actions, dtype = int)
	super_rewards = np.array(super_rewards)
	return super_states, super_actions, super_rewards
	

super_states =  np.empty((0,len_game,observation_space), dtype = int)	# this can be deleted - we will see how to handle this
super_actions = np.array([], dtype = int)
super_rewards = np.array([])
sessgen_time = 0
fit_time = 0
score_time = 0



myRand = random.randint(0,1000) #used in the filename

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
	states_batch = np.transpose(states_batch,axes=[0,2,1]) # this will not be necessary for sequentially generated sessions
	
	states_batch = np.append(states_batch,super_states,axis=0)

	if i>0:
		actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	
	rewards_batch = np.append(rewards_batch,super_rewards)
		
	randomcomp_time = time.time()-tic 
	tic = time.time()

	# elite_actions is a numpy array
	elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
	select1_time = time.time()-tic

	tic = time.time()
	super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, percentile=super_percentile) #pick the sessions to survive
	select2_time = time.time()-tic
	
	tic = time.time()
	super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i]) for i in range(len(super_sessions[2]))]
	super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
	select3_time = time.time()-tic

	#one hot encoding for elite actions
	elite_actions_one_hot = np.zeros((elite_actions.size, 100))
	elite_actions_one_hot[np.arange(elite_actions.size), elite_actions] = 1
	
	tic = time.time()
	model.fit(elite_states, elite_actions_one_hot) #learn from the elite sessions
	fit_time = time.time()-tic
	
	tic = time.time()
	
	super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
	super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
	super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
	
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

	'''if (i%50 == 0):	# Make a plot of best generation every 50th iteration
		max_idx = np.argmax(rewards_batch)
		rectangles = rectsFromState(states_batch[max_idx])
		plot_rectangles(rectangles, rewards_batch[max_idx], i)'''

