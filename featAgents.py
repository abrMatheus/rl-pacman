from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util
from feat_utils import NFEATURES, computeDistances, filterDist
import models
import json
from collections import deque

from keras.models import model_from_json, clone_model


gamma_range    = np.array([0.9])
alfa_range     = np.array([0.001, 0.0001])
training_range = np.array([500, 2000, 4000, 10000])
lambda_range   = np.array([0.01,0.1, 0.2])
e_range       = np.array([0.1, 0.3, 0.4])

# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 101 -x 100 -a alfa=0.0001,discount_factor=0.9,Slambda=0.0

# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 201 -x 200 -a alfa=0.0001,discount_factor=0.9,epsilon=0.4,Slambda=0.1 -q

class FeatSARSAAgent(Agent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=1000, discount_factor=0.9, Slambda=0.1, epsilon=0.1, log_name=None):

        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST) #, Directions.STOP)
        

        self.discount_factor = float(discount_factor)
        self.alfa = float(alfa)
        self.lambd = float(Slambda)

        #self.i_epsilon = 1 if self.alfa > 0 else 0.0
        self.epsilon   = float(epsilon)

        self.last_state = None
        self.is_train = True
        self.total_reward = 0
        self.num_actions = 0
        self.ngames = 0
        self.maxa = maxa

        self.last_feat = None

        #main weights
        self.weights  = None
        self.curr_feat = None

        self.gamma_num_range = gamma_range
        self.e_num_range    = e_range
        self.lambda_num_range = lambda_range
        self.training_number_range = training_range
        self.alfa_num_range = alfa_range

        self.log_name = log_name
        self.rlog = []
        self.slog = []


    def update_log(self, state):
            self.rlog.append(self.total_reward)
            self.slog.append(state.getScore())

            tmp = {
                'score' : self.slog,
                'reward': self.rlog
            }

            with open(self.log_name, "wb") as handle:
                pickle.dump(tmp, handle)

    def init_weights(self):

        self.weights = np.zeros(NFEATURES+1)
        self.ztrace  = np.zeros(NFEATURES+1)
        self.Q = self.weights

    def saveTable(self, filename):
        tdict = {'weights' : self.weights}

        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    def final(self, state):

        if self.is_train:
            if state.isLose():
                self.reward = -500
            else:
                self.reward = 5000
            
            best_a, currQ = self.greedyAction(state)
            self.updateWeights(state, terminal=True)

            if self.log_name is not None:
                self.update_log(state)

        #print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epsilon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1
        self.ztrace[:] = 0

    def getAction(self, state):

        if self.weights is None:
            self.init_weights()
    
        if self.num_actions > self.maxa: #max number of actions
            state.data._lose = True
            return 'Stop'

        if not self.is_train:
            self.best_a, currQ = self.greedyAction(state)

        elif self.last_feat is None:
            #fist action
            self.best_a = self.randomAction(state)
            num_food = state.getNumFood()
            dists = computeDistances(state, num_food, self.directions)
            dists = filterDist(dists)
            self.curr_feat = self.featArray(dists)

        else:
            self.best_a, currQ = self.egreedyAction(state)
            self.reward = self.getReward(state)
            self.updateWeights(state)


        self.last_state = state
        self.num_actions += 1
        self.last_feat = self.curr_feat
        return self.best_a
    

    def updateWeights(self, state, terminal=False):

        oldQ = self.computeScoreFromFeat(self.last_feat)
        currQ = 0
        if not terminal:
            dists = computeDistances(state, state.getNumFood(), self.directions)
            dists = filterDist(dists)
            self.curr_feat = self.featArray(dists)
            currQ = self.computeScoreFromFeat(self.curr_feat)
        
        
        td_error = self.reward + self.discount_factor*currQ - oldQ

        self.ztrace = self.discount_factor*self.lambd*self.ztrace
        temp_sum = (self.ztrace*self.curr_feat).sum()
        self.ztrace += (1 - self.alfa*self.discount_factor*self.lambd*temp_sum)*self.curr_feat
        

        #self.weights +=  self.alfa*( (td_error - oldQ)*curr_feat)
        self.weights += (self.alfa * td_error * self.curr_feat)*self.ztrace - self.alfa*(currQ-oldQ)*self.curr_feat

        self.total_reward += self.reward




    def randomAction(self,state):
        legal_actions = state.getLegalActions()
        action = random.choice(legal_actions)

        return action


    def greedyAction(self, state):
        legal = state.getLegalActions()
        try:
            legal.remove(Directions.STOP)
        except ValueError:
            pass

        successors = [(state, 'Stop')]
        for action in legal:
            successor = state.generateSuccessor(0, action)
            successors.append((successor, action))

        num_food = state.getNumFood()
        num_capsules = len(state.getCapsules())

        scored = []
        for state, action in successors:
            dists = computeDistances(state, num_food, self.directions)
            dists = filterDist(dists)
            score = self.computeScoreFromDist(dists)
            scored.append((score, action))

        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]

        return random.choice(bestActions), bestScore

    def egreedyAction(self, state):
        p = random.random()

        if p > self.epsilon:
            action, score =  self.greedyAction(state)
            
        else:
            action = random.choice(state.getLegalActions())

            successor = state.generateSuccessor(0, action)
            num_food = successor.getNumFood()
            dists = computeDistances(successor, num_food, self.directions)
            dists = filterDist(dists)
            score = self.computeScoreFromDist(dists)


        #self.epsilon = self.i_epsilon/(.2*self.ngames+1)
        #self.epsilon  = self.i_epsilon/np.log(self.time+0.1)
        return action, score



    def getReward(self, state):

        reward = -1 

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 100

        if len(state.getCapsules()) < len(self.last_state.getCapsules()):
            reward += 5000        

        # reward if ghosts are eaten
        if sum(state.data._eaten) > 0:
            reward += 2500

        if state.isWin():
            reward += 5000

        if len(state.getCapsules()) < len(self.last_state.getCapsules()):
            reward += 100

        if sum(state.data._eaten) > 0:
            reward += 250

        elif state.isLose():
            reward += -500

        return reward

    def getReward_(self, state):

        reward = -1

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 500

        if state.isWin():
            reward += 50000

        if len(state.getCapsules()) < len(self.last_state.getCapsules()):
            reward += 250

        if sum(state.data._eaten) > 0:
            reward += 750

        elif state.isLose():
            reward += -5000

        return reward

    def computeScoreFromDist(self, dists):
        feature = self.featArray(dists)
        score = self.computeScoreFromFeat(feature)
        return score

    def computeScoreFromFeat(self, feature):
        score = (feature*self.weights).sum()
        return score

    def featArray(self, dist):

        features = np.zeros([NFEATURES +1])

        features[0] = dist['food']                                        #1-distance to fodd
        if (dist['ghost']<3): features[1] = 1.0                           #2-have a ghost nearby
        #features[2] = 1/(dist['ghost']+0.001)                            #3-inverse distance to g
        features[2]  = dist['ghost']                                      #3-distance to ghost
        if (dist['ghost']<3 and dist['food']<2) : features [3] = 1.0      #4-ghost and food nearby
    
        features[-1] = 1                                                  #bias
        
        features = features/np.abs(features).sum()                        #normalize features 
        return features

    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):
        l_size  = self.lambda_num_range.shape[0]
        e_size  = self.e_num_range.shape[0]
        a_size   = self.alfa_num_range.shape[0]
        g_size   = self.gamma_num_range.shape[0]
        ntr_size = self.training_number_range.shape[0]

        ntr = i%ntr_size
        g   = (i//ntr_size)%g_size
        a   = i//(g_size*ntr_size)%a_size
        e  = i//(a_size*g_size*ntr_size)%e_size
        l  = i//(e_size*a_size*g_size*ntr_size)%l_size

        return self.lambda_num_range[l], self.e_num_range[e], self.alfa_num_range[a], self.gamma_num_range[g], self.training_number_range[ntr]

    def get_param_names(self):
        return ['Lambda', 'Epsilon', 'Alpha', 'Gamma', 'N']

    def set_parameters_in_iteration(self, i):
        l, e, a, g, ntr = self.get_parameters_in_iteration(i)
        self.lambd = l
        self.epsilon = e
        self.alfa = a
        self.discount_factor = g

        return ntr

    def get_training_space_size(self):
        l   = self.lambda_num_range.shape[0]
        e   = self.e_num_range.shape[0]
        a   = self.alfa_num_range.shape[0]
        g   = self.gamma_num_range.shape[0]
        ntr = self.training_number_range.shape[0]
        

        return l*e*a*g*ntr

    def reset_tables(self):
        self.weights[:] = 0
        self.ztrace[:] = 0
        #self.episode_size = 0
        #self.episode = []
        self.reward=0
        self.last_state = None
        self.last_feat = None
        self.num_actions = 0
        self.total_reward = 0

    def write_best_parameters(self, best_parameters, average_score):
        best_l = best_parameters[0]
        best_e = best_parameters[1]
        best_alfa = best_parameters[2]
        best_gamma = best_parameters[3]
        best_n_training = best_parameters[4]
        print("Lambda:", best_l)
        print("E : ", best_e)
        print("Alfa : ", best_alfa)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    #----------------------------------------------------------------------------



#python pacman.py -l customMaze -p FeatQLAgent -g PatrolGhost -n 201 -x 200 -a alfa=0.0001,discount_factor=0.9,epsilon=0.3 -q
class FeatQLAgent(FeatSARSAAgent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=3000, discount_factor=0.9, epsilon=0.1, log_name=None):

        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST) #, Directions.STOP)
        

        self.discount_factor = float(discount_factor)
        self.alfa = float(alfa)

        #self.i_epsilon = 1 if self.alfa > 0 else 0.0
        self.epsilon   = float(epsilon)

        self.last_state = None
        self.is_train = True
        self.total_reward = 0
        self.num_actions = 0
        self.ngames = 0
        self.maxa=maxa
        
        self.curr_feat = None
        self.last_feat = None

        #main weights
        self.weights  = None

        self.gamma_num_range = gamma_range
        self.e_num_range    = e_range
        self.training_number_range = training_range
        self.alfa_num_range = alfa_range

        self.log_name = log_name
        self.rlog = []
        self.slog = []


    def update_log(self, state):
            self.rlog.append(self.total_reward)
            self.slog.append(state.getScore())

            tmp = {
                'score' : self.slog,
                'reward': self.rlog
            }

            with open(self.log_name, "wb") as handle:
                pickle.dump(tmp, handle)
    


    def saveTable(self, filename):
        tdict = {'weights' : self.weights}

        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    def final(self, state):
        if self.is_train:
            if state.isLose():
                self.reward = -500
            else:
                self.reward = 5000
            
            best_a, currQ = self.greedyAction(state)
            self.updateWeights(state, terminal=True)

            if self.log_name is not None:
                self.update_log(state)

        #print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epsilon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1

    def init_weights(self):
        self.weights = np.zeros(NFEATURES+1)
        #self.weights  = np.random.randn(NFEATURES+1)
        self.Q = self.weights

    def updateWeights(self, state, terminal=False):

        oldQ = self.computeScoreFromFeat(self.last_feat)

        dists = computeDistances(state, state.getNumFood(), self.directions)
        dists = filterDist(dists)
        self.curr_feat = self.featArray(dists)
        #currQ = self.computeScoreFromFeat(self.curr_feat)

        currQ = 0

        if not terminal:
            ###get the best score (or currQ)
            _, currQ = self.greedyAction(state)
        
        td_error = self.reward + self.discount_factor*currQ - oldQ

        #self.weights +=  self.alfa*( (td_error - oldQ)*curr_feat)
        self.weights += self.alfa * td_error * self.curr_feat

        self.total_reward += self.reward

    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):
        e_size  = self.e_num_range.shape[0]
        a_size   = self.alfa_num_range.shape[0]
        g_size   = self.gamma_num_range.shape[0]
        ntr_size = self.training_number_range.shape[0]

        ntr = i%ntr_size
        g   = (i//ntr_size)%g_size
        a   = i//(g_size*ntr_size)%a_size
        e  = i//(a_size*g_size*ntr_size)%e_size

        return self.e_num_range[e], self.alfa_num_range[a], self.gamma_num_range[g], self.training_number_range[ntr]

    def get_param_names(self):
        return ['Epsilon', 'Alpha', 'Gamma', 'N']

    def set_parameters_in_iteration(self, i):
        e, a, g, ntr = self.get_parameters_in_iteration(i)
        self.epsilon = e
        self.alfa = a
        self.discount_factor = g

        return ntr

    def get_training_space_size(self):
        e  = self.e_num_range.shape[0]
        a   = self.alfa_num_range.shape[0]
        g   = self.gamma_num_range.shape[0]
        ntr = self.training_number_range.shape[0]
        

        return e*a*g*ntr

    def reset_tables(self):
        self.weights[:] = 0
        #self.episode_size = 0
        #self.episode = []
        self.reward=0
        self.last_state = None
        self.last_feat = None
        self.num_actions = 0
        self.total_reward = 0

    def write_best_parameters(self, best_parameters, average_score):
        best_e = best_parameters[0]
        best_alfa = best_parameters[1]
        best_gamma = best_parameters[2]
        best_n_training = best_parameters[3]
        print("E : ", best_e)
        print("Alfa : ", best_alfa)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    #----------------------------------------------------------------------------



class DQNAgent(FeatQLAgent):
<<<<<<< HEAD
    "Function approximation SARSA"
    def __init__ (self, maxa=1000, epsilon=1, log_name=None, gamma = 0.99, pretrained_model=None, output_model="./model.h5", map_size="7.20.3"):
=======
    "Deep Q-Network"
    def __init__ (self, maxa=1000, epsilon=1, log_name=None, gamma = 0.99, pretrained_model=None, output_model="./model.h5"):
>>>>>>> 7dff2124a3650aaa9f41a44f745840a3af2d9cd3

        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST) #, Directions.STOP)



        #self.i_epsilon = 1 if self.alfa > 0 else 0.0
        self.epsilon   = float(epsilon)
        map_size_array = map_size.split(".")
        self.map_size = (int(map_size_array[0]),int(map_size_array[1]),int(map_size_array[2]))

        self.last_state = None
        self.is_train = True
        self.total_reward = 0
        self.num_actions = 0
        self.maxa=maxa
        self.model_output_path = output_model

        self.batch_size_num_range = np.array([8,16,32])
        self.network_update_freq_num_range = np.array([250,500,1000])
        self.gamma = gamma

        self.log_name = log_name
        self.rlog = []
        self.slog = []


        self.last_action = "Stop"
        self.last_map = None
        self.batch_size = 32
        self.replay_memory = 100000
        self.min_observation_size = 5000
        self.t = 0
        self.loss = 0
        self.network_update_freq = 500
        self.save_frequency = 500
        self.final_epsilon = 0.01
        self.explore = 10000
        self.reward = 0
        self.episode_number = 0

        self.loss_history = []
        self.reward_history = []

        self.history = deque()
        if pretrained_model is None:
            self.model = models.createCNNwithAdam(inputDimensions=self.map_size)
        else:
            self.model = models.createCNNwithAdam(pretrained=pretrained_model, inputDimensions=self.map_size)

        self.targetModel = clone_model(self.model)
        self.targetModel.set_weights(self.model.get_weights())

    def update_log(self, state):
            self.rlog.append(self.total_reward)
            self.slog.append(state.getScore())

            tmp = {
                'score' : self.slog,
                'reward': self.rlog
            }

            with open(self.log_name, "wb") as handle:
                pickle.dump(tmp, handle)

    def saveTable(self, filename):
        tdict = {'weights' : self.weights}

        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)


    def greedyAction(self, state):
        legal = state.getLegalActions()
        legal_vector = np.zeros(5)
        all_actions = [Directions.STOP, Directions.NORTH, Directions.SOUTH, Directions.WEST, Directions.EAST]
        for i in range(legal_vector.size):
            legal_vector[i] = 0 if all_actions[i] in legal else -10000
        current_map = util.get_state_image(state, self.last_map)
        q = self.model.predict(current_map)
        
        sorted_ix = np.argsort(q)
        sorted_ix = np.squeeze(sorted_ix)
        sorted_ix = sorted_ix[::-1]

        for i in range(len(sorted_ix)):
            action = all_actions[sorted_ix[i]]
            if action in legal:
                return action


    def egreedyAction(self, state):
        p = random.random()

        if p > self.epsilon:
            action =  self.greedyAction(state)

        else:
            action = random.choice(state.getLegalActions())

            successor = state.generateSuccessor(0, action)
            num_food = successor.getNumFood()
            dists = computeDistances(successor, num_food, self.directions)
            dists = filterDist(dists)

        return action


    def getAction(self, state):
        self.t = self.t + 1

        if self.num_actions > self.maxa: #max number of actions
            state.data._lose = True
            return 'Stop'

        if not self.is_train:
            self.best_a = self.greedyAction(state)

        elif self.last_map is None:
            #first action
            self.best_a = self.randomAction(state)
            num_food = state.getNumFood()
            dists = computeDistances(state, num_food, self.directions)
            dists = filterDist(dists)
            self.curr_feat = self.featArray(dists)
            self.reward = 0

        else:
            self.best_a = self.egreedyAction(state)
            self.reward = self.getReward(state)
            #if self.last_action == "Stop":
            #    self.reward+=-100
            #self.updateWeights(state)

        current_map = util.get_state_image(state, self.last_map)
        if self.last_map is not None:
            if len(self.history) > self.replay_memory:
                self.history.popleft()
            self.history.append((self.last_map, util.get_action_index(self.last_action), self.reward, current_map, False))
        self.last_action = self.best_a
        self.last_map = current_map

        self.last_state = state
        self.num_actions += 1

        #TRAINING
        if self.t >= self.min_observation_size and self.is_train:
            self.perform_train_pass()

        if self.epsilon > self.final_epsilon and self.t > self.min_observation_size:
            self.epsilon -= (self.epsilon - self.final_epsilon) / self.explore

        return self.best_a

    def final(self, state):
        if self.is_train:
            # if state.isLose():
            #     self.reward = -500
            # else:
            #     self.reward = 5000
            self.reward = self.getReward(state)
            self.reward_history.append(state.getScore())

            current_map = util.get_state_image(state, self.last_map)
            if self.last_map is not None:
                if len(self.history) > self.replay_memory:
                    self.history.popleft()
                self.history.append((self.last_map, util.get_action_index(self.last_action), self.reward, current_map, True))
            self.last_action = self.best_a

            self.last_state = state
            self.num_actions += 1

            if self.t >= self.min_observation_size:
                self.perform_train_pass()

            if self.log_name is not None:
                self.update_log(state)
        #print("Score: ", state.getScore(), state.getNumFood())
        self.episode_number+=1

        if self.episode_number > 900:
            util.plot_reward_history(self.reward_history)

        #print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epsilon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.last_map = None

    def perform_train_pass(self):
        minibatch = random.sample(self.history, self.batch_size)
        prev_map, action, reward, cur_map, is_terminal = zip(*minibatch)
        prev_map = np.concatenate(prev_map)
        cur_map = np.concatenate(cur_map)
        y_j = self.targetModel.predict(prev_map)
        Q_sa = self.targetModel.predict(cur_map)
        y_j[range(self.batch_size), action] = reward + self.gamma*np.max(Q_sa, axis=1)*np.invert(is_terminal)
        self.loss = self.model.train_on_batch(prev_map, y_j)
        #print("Loss", self.loss)

        #if self.t > 500:
        self.loss_history.append(self.loss)

        if self.t % self.network_update_freq == 0:
            self.targetModel = clone_model(self.model)
            self.targetModel.set_weights(self.model.get_weights())

        if self.t % self.save_frequency == 0:
            #print((self.t/self.save_frequency), "Done, Loss = ", self.loss, "Epsilon = ", self.epsilon)
            self.model.save_weights(self.model_output_path, overwrite=True)
            with open("./model.json", "w") as outfile:
                json.dump(self.model.to_json(), outfile)
        #if self.t%1000 == 0:
        #    util.plot_loss_history(self.loss_history)




    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):

        self.batch_size_num_range = np.array([4, 8, 32])
        self.network_update_freq_num_range = np.array([500])

        b_size  = self.batch_size_num_range.shape[0]
        r_size   = self.network_update_freq_num_range.shape[0]

        r = i%r_size
        b = (i//r_size)%b_size

        return self.network_update_freq_num_range[r], self.batch_size_num_range[b]

    def get_param_names(self):
        return ['Network Update Frequency', 'Batch Size']

    def set_parameters_in_iteration(self, i):
        r, b = self.get_parameters_in_iteration(i)
        self.network_update_freq = r
        self.batch_size = b

        return 1000

    def get_training_space_size(self):
        r   = self.network_update_freq_num_range.shape[0]
        b   = self.batch_size_num_range.shape[0]
        ntr = 1


        return r*b*ntr

    def reset_tables(self):
        self.epsilon = float(1.0)
        self.model = models.createCNNwithAdam(inputDimensions=self.map_size)
        self.targetModel = clone_model(self.model)
        self.targetModel.set_weights(self.model.get_weights())
        self.reward=0
        self.last_state = None
        self.last_feat = None
        self.last_action = "Stop"
        self.last_map = None
        self.num_actions = 0
        self.total_reward = 0
        self.history = deque()
        self.t = 0
        self.loss = 0

    def write_best_parameters(self, best_parameters, average_score):
        best_r = best_parameters[0]
        best_b = best_parameters[1]
        best_n_training = 1000
        print("Network Update Frequency : ", best_r)
        print("Batch Size : ", best_b)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    #----------------------------------------------------------------------------

