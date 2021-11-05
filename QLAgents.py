from game import Agent
from game import Directions
import util
import json
import numpy as np

class QLAgent(Agent):


    def __init__(self, path='rl-data/', alpha=0.2, gamma=0.9, max_steps=5000, total_episodes=10):

        self.Q = dict()
                
        self.total_episodes = total_episodes
        self.elapsed_episodes = 0

        self.max_steps = max_steps
        self.elapsed_steps = 0

        self.eps_a = .5
        self.eps_b = .1
        self.eps_c = .1
        self.epsilon = self.getEpsilon(self.elapsed_episodes)

        self.alpha = alpha
        self.gamma = gamma
        
        self.previous_state = None
        self.previous_action = None


    def updateQ(self, state):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_a_prime = self.greedyPolicy(state)

        self.Q[s][a] += self.alpha * (reward + self.gamma * self.Q[s_prime][max_a_prime] - self.Q[s][a])


    def greedyPolicy(self, state):
        # getting state_id
        state_id = util.get_state_id(state)

        action_values = self.Q[state_id]
        
        action = max(action_values, key=action_values.get)

        return action


    def egreedyPolicy(self, state, legal_actions):

        if util.flipCoin(self.epsilon):
            # random action
            action = np.random.choice(legal_actions)
        else:
            # greedy action
            action = self.greedyPolicy(state)

        return action
            

    def getAction(self, state):

        self.elapsed_steps += 1
        if self.elapsed_steps > self.max_steps:
            state.data._lose = True
            print('Stuck')
            return Directions.STOP

        state_id = util.get_state_id(state)

        legal_actions = state.getLegalActions()
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        # if state is not in Q
        if(state_id not in self.Q):
            self.Q[state_id] = dict()
            for action in legal_actions:
                self.Q[state_id][action] = 0

        if self.previous_state is not None:
            self.updateQ(state)

        action = self.egreedyPolicy(state, legal_actions)

        self.previous_action = action
        self.previous_state = state

        return action


    def getReward(self, state):        
        # standard reward
        reward = -1 
        # reward for eating food
        if state.getNumFood() < self.previous_state.getNumFood():
            reward = 10
        # reward for winning
        if state.isWin():
            reward = 5000
        # reward for losing
        elif state.isLose():
            reward = -500

        return reward


    def final(self, state):

        self.elapsed_episodes += 1
        self.elapsed_steps = 0
        self.epsilon = self.getEpsilon(self.elapsed_episodes)

        state_id = util.get_state_id(state)
        if state_id not in self.Q:
            reward = self.getReward(state)
            
            s = util.get_state_id(self.previous_state)
            a = self.previous_action
            
            self.Q[s][a] += self.alpha * (reward - self.Q[s][a])
        else:
            self.updateQ(state)

        self.previous_state = None
        self.previous_action = None


    def getEpsilon(self, elapsed_episodes):

        std_elapsed_episodes = (elapsed_episodes - self.eps_a * self.total_episodes) \
                            / (self.eps_b * self.total_episodes)
        cosh = np.cosh(np.exp(-std_elapsed_episodes))
        epsilon = 1.1 - (1 / cosh + (elapsed_episodes * self.eps_c / self.total_episodes))

        return epsilon
        