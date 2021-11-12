from game import Agent
from game import Directions
import util
import json
import numpy as np
import os

class DoubleQLAgent(Agent):

    gamma_range    = np.array([0.9, 0.99])
    alpha_range     = np.array([0.1, 0.2])
    training_range = np.array([1000, 2500, 5000])

    def __init__(self, path='dql_output', alpha=0.2, gamma=0.9, max_steps=5000, total_episodes=10, heatmap='qa'):

        self.Q_A = dict()
        self.Q_B = dict()
                
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
        self.heatmap = heatmap

    def updateQ_A(self, state, heatmap):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_a_prime = self.greedyPolicy(state, self.Q_A)

        self.Q_A[s][a] += self.alpha * (reward + self.gamma * self.Q_B[s_prime][max_a_prime] - self.Q_A[s][a])

        if self.heatmap == "qa" and util.max_Q_val is not None:
            util.update_heatmap(s, a, self.Q_A[s][a])

    def updateQ_B(self, state, heatmap):
        # getting reward
        reward = self.getReward(state)

        s = util.get_state_id(self.previous_state)
        a = self.previous_action

        s_prime = util.get_state_id(state)
        max_b_prime = self.greedyPolicy(state, self.Q_B)

        self.Q_B[s][a] += self.alpha * (reward + self.gamma * self.Q_A[s_prime][max_b_prime] - self.Q_B[s][a])

        if self.heatmap == "qb" and util.max_Q_val is not None:
            util.update_heatmap(s, a, self.Q_B[s][a])

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

        if self.is_train and self.previous_state is not None:
            if util.flipCoin(0.5):
                self.updateQ_A(state, self.heatmap)
            else:
                self.updateQ_B(state, self.heatmap)

        if self.is_train:
            action = self.egreedyPolicyDoubleQL(state, legal_actions)
        else:
            action = self.greedyPolicyDoubleQL(state)

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

        if self.is_train:
            state_id = util.get_state_id(state)
            if state_id not in self.Q_A:
                reward = self.getReward(state)
                
                s = util.get_state_id(self.previous_state)
                a = self.previous_action

                if util.flipCoin(0.5):
                    self.Q_A[s][a] += self.alpha * (reward - self.Q_A[s][a])
                else:
                    self.Q_B[s][a] += self.alpha * (reward - self.Q_B[s][a])
            else:
                if util.flipCoin(0.5):
                    self.updateQ_A(state)
                else:
                    self.updateQ_B(state)

        self.previous_state = None
        self.previous_action = None

        if self.is_train:
            self.scores.append(state.getScore())
        else:
            filename = "dql_scores.npy"
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
        
        dicts = dict()
        dicts['Q_A'] = self.Q_A
        dicts['Q_B'] = self.Q_B

        with open(output_model_path,"w") as file_out:
            
            json.dump(dicts, file_out)        


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
        del self.Q_A
        del self.Q_B
        self.Q_A = dict()
        self.Q_B = dict()
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