import numpy as np
import matplotlib.pyplot as plt

from game import Game

# define actions
actions = ('left', 'down', 'right', 'up')


def print_policy():
    policy = Q.argmax(axis=1)
    policy = np.asarray([actions[action] for action in policy])
    policy = policy.reshape((game.max_row, game.max_col))
    print("\n\n".join('\t'.join(line) for line in policy)+"\n")


# Init Game Instance
game = Game(living_penalty=-0.04, render=False)

# Initialize Q-table with all zeros (state_space, action_space)
# TODO Your code here (one line)
state_space = 64
action_space = 4
Q = np.zeros((state_space, action_space))

# Set learning parameters
lr = .8  # learning rate
y = .95  # discount factor
num_episodes = 2000

# create lists to contain total rewards and per episode
jList = []
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = game.reset()
    rAll = 0
    j = 0

    # The Q-Table learning algorithm
    while j < 100:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,action_space)*(1./(i+1)))

        # Get new state and reward from environment
        r, s1, game_over = game.perform_action(actions[a])
        # print(j, s1)

        # Update Q-Table with new knowledge implementing the Bellmann Equation
        Q[s,a] = Q[s,a] + lr* (r + y * (np.max(Q[s1,:]) - Q[s,a]) )

        # Add reward to list
        rAll += r

        # Replace old state with new
        s = s1

        if game_over:
            break
    jList.append(j)
    rList.append(rAll)

print("\nScore over time: " + str(sum(rList)/num_episodes))
print("\nFinal Q-Table Policy:\n")
# print_policy()
plt.plot(rList)
# plt.plot(jList)
plt.show()
