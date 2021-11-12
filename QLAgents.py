from game import Agent
from game import Directions
import util
import json
import numpy as np
import os

class QLAgent(Agent):

    gamma_range    = np.array([0.9, 0.99])
    alpha_range     = np.array([0.1, 0.2])
    training_range = np.array([1000, 2500, 5000])

    def __init__(self, path='ql_output', alpha=0.1, gamma=0.99, max_steps=5000, total_episodes=2500):

        self.Q = dict()
        
        self.total_episodes = int(total_episodes)
        self.elapsed_episodes = 0

        self.max_steps = int(max_steps)
        self.elapsed_steps = 0

        self.eps_a = .5
        self.eps_b = .1
        self.eps_c = .1
        self.epsilon = self.getEpsilon(self.elapsed_episodes)

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        
        self.previous_state = None
        self.previous_action = None

        self.is_train = False
        self.scores = []

        self.path = path;

    def updateQ(self, state):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_a_prime = self.greedyPolicy(state)

        self.Q[s][a] += self.alpha * (reward + self.gamma * self.Q[s_prime][max_a_prime] - self.Q[s][a])

        if(util.max_Q_val is not None):
            util.update_heatmap(s, a, self.Q[s][a])

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

        if self.is_train and self.previous_state is not None:
            self.updateQ(state)

        if self.is_train:
            action = self.egreedyPolicy(state, legal_actions)
        else:
            action = self.greedyPolicy(state)

        self.previous_action = action
        self.previous_state = state

        return action

    def getReward(self, state):        
        # standard reward
        reward = -1 
        # reward for eating food
        if state.getNumFood() < self.previous_state.getNumFood():
            reward += 10
        # reward for winning
        if state.isWin():
            reward += 5000
        # reward for losing
        elif state.isLose():
            reward += -500

        return reward

    def final(self, state):

        self.elapsed_episodes += 1
        self.elapsed_steps = 0
        self.epsilon = self.getEpsilon(self.elapsed_episodes)

        if self.is_train:
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

        if self.is_train:
            self.scores.append(state.getScore())
        else:
            filename = "ql_scores.npy"
            # create the path if it doesn't exist
            if not os.path.isdir(self.path):
                os.makedirs(self.path)
            filepath = os.path.join(self.path, filename)
            # create the scores file only if it doesn't exist
            if not os.path.isfile(filepath):
                np.save(filepath, self.scores)

    def getEpsilon(self, elapsed_episodes):

        std_elapsed_episodes = (elapsed_episodes - self.eps_a * self.total_episodes) \
                            / (self.eps_b * self.total_episodes)
        cosh = np.cosh(np.exp(-std_elapsed_episodes))
        epsilon = 1.1 - (1 / cosh + (elapsed_episodes * self.eps_c / self.total_episodes))

        return epsilon

    def saveTable(self, output_model_path):
        
        with open(output_model_path,"w") as file_out:
            
            json.dump(self.Q, file_out)

    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):
        x_size = self.alpha_range.shape[0]
        y_size = self.gamma_range.shape[0]
        z_size = self.training_range.shape[0]

        z = i%z_size
        y = (i//z_size)%y_size
        x = i//(y_size*z_size)%x_size

        return self.alpha_range[x], self.gamma_range[y], self.training_range[z]

    def set_parameters_in_iteration(self, i):
        x, y, z = self.get_parameters_in_iteration(i)
        self.alpha = x
        self.gamma = y

        return z

    def get_training_space_size(self):
        x = self.alpha_range.shape[0]
        y = self.gamma_range.shape[0]
        z = self.training_range.shape[0]

        return x*y*z

    def reset_tables(self):
        del self.Q
        self.Q = dict()
        self.elapsed_steps = 0

    def write_best_parameters(self, best_parameters, average_score):
        best_alpha = best_parameters[0]
        best_gamma = best_parameters[1]
        best_n_training = best_parameters[2]
        print("Alpha : ", best_alpha)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    def get_param_names(self):
        return ['Alpha', 'Gamma', 'N']
    #----------------------------------------------------------------------------