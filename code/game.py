import numpy as np
import sys
import os
from util.getkey import get_manual_arrow_key

# define the map
MAPS = {
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

# define some colors for terminal output
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"


class Game():
    '''
    The task of the game is to reach the goal without falling into a hole.
    Therefore, the player can securely move on frozen surfaces but has to avoid the holes.
    The surface is described using a grid like the following:

        SFFFFFFF
        FFFFFFFF
        FFFHFFFF
        FFFFFHFF
        FFFHFFFF
        FHHFFFHF
        FHFFHFHF
        FFFHFFFG

    S : starting point, safe    
    F : frozen surface, safe    
    H : hole, fall to your doom 
    G : goal    

    The game ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, -1 if you fall in a hole and the living_penalty=zero otherwise.

    Attributes:
        living_penalty -- defines the reward of the frozen surfaces
        map_size -- defines the size of the game map
        render -- (boolean) whether to render the game or not
    '''

    def __init__(self, living_penalty=0, map_size='8x8', render=True):
        map_layout = np.asarray(MAPS[map_size], dtype='c')
        self.max_row, self.max_col = map_layout.shape

        map_layout = map_layout.tolist()
        self.map_layout = [[c.decode('utf-8') for c in line] for line in map_layout]
        self.x_pos, self.y_pos = 0, 0
        self.lastaction = None
        self.game_over = False
        self.render = render
        self.living_penalty = living_penalty

        if self.render:
            self.rendering()

    def reset(self):
        '''This method resets the game.'''
        self.x_pos, self.y_pos = 0, 0
        self.lastaction = None
        self.game_over = False
        return self.get_state()

    def get_reward(self):
        '''Returns the reward of the current player position and sets the game_over boolean.'''
        label = self.map_layout[self.x_pos][self.y_pos]
        if label in 'FS':
            return self.living_penalty
        if label in 'H':
            self.game_over = True
            if self.render:
                sys.stdout.write(RED)
                sys.stdout.write('\n Agent walked into a hole! Game Over!')
            return -1
        if label in 'G':
            self.game_over = True
            if self.render:
                sys.stdout.write('\n Agent walked into the goal! You won the Game!')
            return 1

    def get_state(self):
        '''Returns the current state of the player'''
        return (self.x_pos * 8) + self.y_pos 

    def perform_action(self, action):
        '''
        Performs the given action in the environment. 
        
        action -- String representation of the action, e.g. left, down, right or up.
        '''
        if action == 'left':
            self.x_pos = max(self.x_pos, 0)
            self.y_pos = max(self.y_pos-1, 0)
        elif action == 'down':
            self.x_pos = min(self.x_pos+1, self.max_row-1)
            self.y_pos = max(self.y_pos, 0)
        elif action == 'right':
            self.x_pos = max(self.x_pos, 0)
            self.y_pos = min(self.y_pos+1, self.max_col-1)
        elif action == 'up':
            self.x_pos = max(self.x_pos-1, 0)
            self.y_pos = max(self.y_pos, 0)
        self.lastaction = action
        if self.render:
            self.rendering()

        return self.get_reward(), self.get_state(), self.game_over

    def rendering(self):
        '''This method renders the game in the console. For each step the console will be cleared and then reprinted'''
        os.system('cls||clear')
        if self.lastaction is not None:
            sys.stdout.write(BLUE)
            sys.stdout.write("  ({})\n".format(self.lastaction))
        else:
            sys.stdout.write("\n")

        for idx_r, r in enumerate(self.map_layout):
            for idx_c, c in enumerate(r):
                sys.stdout.write(RESET)
                if c in 'H':
                    sys.stdout.write(RED)
                if c in 'G':
                    sys.stdout.write(GREEN)
                if idx_r == self.x_pos and idx_c == self.y_pos:
                    sys.stdout.write(CYAN)
                sys.stdout.write(c + ' ')
            sys.stdout.write('\n')


if __name__ == "__main__":
    game = Game()
    game = Game()
    game_over = False
    while not game_over:
        # play game with helper class, which will detect keyboard inputs
        _, _, g = game.perform_action(get_manual_arrow_key())
        game_over = g
