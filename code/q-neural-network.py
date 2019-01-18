import numpy as np
import matplotlib.pyplot as plt
from game import Game

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# define actions
actions = ('left', 'down', 'right', 'up')


def print_policy():
    policy = [agent(s).argmax(1)[0].detach().item() for s in range(state_space)]
    policy = np.asarray([actions[action] for action in policy])
    policy = policy.reshape((game.max_row, game.max_col))
    print("\n\n".join('\t'.join(line) for line in policy) + "\n")

# define Q-Network


class QNetwork(nn.Module):

    def __init__(self, state_space, action_space):
        super(QNetwork, self).__init__()
        self.state_space = state_space
        self.hidden_size = state_space

        self.l1 = nn.Linear(in_features=self.state_space, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=action_space)

    def forward(self, x):
        x = self.one_hot_encoding(x)
        out1 = torch.sigmoid(self.l1(x))
        return self.l2(out1) 

    def one_hot_encoding(self, x):
        '''
        One-hot encodes the input data, based on the defined state_space.
        '''
        out_tensor = torch.zeros([1, state_space])
        out_tensor[0][x] = 1
        return out_tensor


# Make use of cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Game Instance
game = Game(living_penalty=-0.07, render=False)

# Define State and Action Space
state_space = game.max_row * game.max_col
action_space = len(actions)

# Set learning parameters
e = 0.1  # epsilon
lr = .03  # learning rate
y = .999  # discount factor
num_episodes = 2000

# create lists to contain total rewards and steps per episode
jList = []
rList = []

# init Q-Network
agent = QNetwork(state_space, action_space).to(device)

# define optimizer and loss
# optimizer = optim.SGD(agent.parameters(), lr=lr)
optimizer = optim.Adam(params=agent.parameters())
criterion = nn.SmoothL1Loss()

for i in range(num_episodes):
    # Reset environment and get first new observation
    s = game.reset()
    rAll = 0
    j = 0

    # The Q-Network learning algorithm
    while j < 99:
        j += 1

        # Choose an action by greedily (with e chance of random action) from the Q-network
        with torch.no_grad():
            # Do a feedforward pass for the current state s to get predicted Q-values
            # for all actions (=> agent(s)) and use the max as action a: max_a Q(s, a)
            a = agent(s).max(1)[1].view(1, 1)  # max(1)[1] returns index of highest value

        # e greedy exploration
        if np.random.rand(1) < e:
            a[0][0] = np.random.randint(1, 4)

        # Get new state and reward from environment
        # perform action to get reward r, next state s1 and game_over flag
        # calculate maximum overall network outputs: max_a’ Q(s1, a’).
        r, s1, game_over = game.perform_action(actions[a])

        # Calculate Q and target Q
        q = agent(s).max(1)[0].view(1, 1)
        q1 = agent(s1).max(1)[0].view(1, 1)

        with torch.no_grad():
            # Set target Q-value for action to: r + y max_a’ Q(s’, a’)
            target_q = r + y * q1

        # print(q, target_q)
        # Calculate loss
        loss = criterion(q, target_q)
        if j == 1 and i % 100 == 0:
            print("loss and reward: ", i, loss, r)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add reward to list
        rAll += r

        # Replace old state with new
        s = s1

        if game_over:
            # Reduce chance of random action as we train the model.
            e = 1. / ((i / 50) + 10)
            break
    rList.append(rAll)
    jList.append(j)

print("\Average steps per episode: " + str(sum(jList) / num_episodes))
print("\nScore over time: " + str(sum(rList) / num_episodes))
print("\nFinal Q-Network Policy:\n")
print_policy()
plt.plot(jList)
# plt.plot(rList)
plt.savefig("j_q_network.png")
plt.show()
