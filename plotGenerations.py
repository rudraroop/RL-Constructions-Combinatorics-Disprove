import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_rectangles(rects, score, idx, min, max):
    
    fig, ax = plt.subplots()
    ax.plot([min,max], [min,max], color='white', linewidth=0)

    for rect in rects:
        ax.add_patch(patches.Rectangle(rect.bottomLeft, rect.topRight[0] - rect.bottomLeft[0], rect.topRight[1] - rect.bottomLeft[1], edgecolor = 'black', fill=False, lw=0.3))
    
    if (score > -100):
        plt.annotate('Killed ' + str(score) +" Rectangles", (0,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')
    
    fig.savefig('plots/rects_'+str(idx)+'.png')
    plt.close(fig)
'''
states = np.zeros([4,4,2], dtype=int)
state_next = np.zeros([4,4])
state_next[0][0] = 1
state_next[0][1] = 2
state_next[0][2] = 3
state_next[0][3] = 4
state_next[1][0] = 5
state_next[1][1] = 6
state_next[1][2] = 7
state_next[1][3] = 8
states[0,:,0] = state_next[0]
states[1,:,0] = state_next[1]
print(state_next)
print(states)
states_batch = np.transpose(states,axes=[0,2,1])
print("After transpose, states are: ")
print(states_batch)
'''