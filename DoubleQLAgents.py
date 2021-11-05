from game import Agent
from game import Directions
import util
import json
import numpy as np

class DoubleQLAgent(Agent):


    def __init__(self, path='rl-data/', alpha=0.2, gamma=0.9, max_steps=5000, total_episodes=10):

        self.Q_A = dict()
        self.Q_B = dict()
                
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


    def updateQ_A(self, state):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_a_prime = self.greedyPolicy(state, self.Q_A)

        self.Q_A[s][a] += self.alpha * (reward + self.gamma * self.Q_B[s_prime][max_a_prime] - self.Q_A[s][a])

    def updateQ_B(self, state):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_b_prime = self.greedyPolicy(state, self.Q_B)

        self.Q_B[s][a] += self.alpha * (reward + self.gamma * self.Q_A[s_prime][max_b_prime] - self.Q_B[s][a])


    def greedyPolicy(self, state, Q_X):
        # getting state_id
        state_id = util.get_state_id(state)

        action_values = Q_X[state_id]
        
        action = max(action_values, key=action_values.get)

        return action


    def greedyPolicyDoubleQL(self, state):
        # getting state_id
        state_id = util.get_state_id(state)

        action_values_A = self.Q_A[state_id]
        action_values_B = self.Q_B[state_id]

        action_A = max(action_values_A, key=action_values_A.get)
        action_B = max(action_values_B, key=action_values_B.get)
        
        if action_values_A[action_A] >= action_values_B[action_B]:
            action = action_A
        else:
            action = action_B

        return action


    def egreedyPolicyDoubleQL(self, state, legal_actions):

        if util.flipCoin(self.epsilon):
            # random action
            action = np.random.choice(legal_actions)
        else:
            # greedy action
            action = self.greedyPolicyDoubleQL(state)

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

        # if it is not in both Q_A and Q_B
        if state_id not in self.Q_A:
            self.Q_A[state_id] = dict()
            self.Q_B[state_id] = dict()

            for action in legal_actions:
                self.Q_A[state_id][action] = 0
                self.Q_B[state_id][action] = 0

        # choose action based on Q_A and Q_B
        action = self.egreedyPolicyDoubleQL(state, legal_actions)

        if self.previous_state is not None:
            if util.flipCoin(0.5):
                self.updateQ_A(state)
            else:
                self.updateQ_B(state)

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
        if state_id not in self.Q_A:
            reward = self.getReward(state)
            
            s = util.get_state_id(self.previous_state)
            a = self.previous_action

            if util.flipCoin(0.5):
                # updating Q_A
                self.Q_A[s][a] += self.alpha * (reward - self.Q_A[s][a])
            else:
                # updating Q_B
                self.Q_B[s][a] += self.alpha * (reward - self.Q_B[s][a])
        else:
            if util.flipCoin(0.5):
                self.updateQ_A(state)
            else:
                self.updateQ_B(state)

        self.previous_state = None
        self.previous_action = None


    def getEpsilon(self, elapsed_episodes):

        std_elapsed_episodes = (elapsed_episodes - self.eps_a * self.total_episodes) \
                            / (self.eps_b * self.total_episodes)
        cosh = np.cosh(np.exp(-std_elapsed_episodes))
        epsilon = 1.1 - (1 / cosh + (elapsed_episodes * self.eps_c / self.total_episodes))

        return epsilon
        