from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util
from feat_utils import NFEATURES, computeDistances, filterDist


gamma_range    = np.array([0.9])
alfa_range     = np.array([0.001, 0.0001])
training_range = np.array([500, 2000, 4000, 10000])
lambda_range   = np.array([0.01,0.1, 0.2])
e_range       = np.array([0.1, 0.3, 0.4])

# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 101 -x 100 -a alfa=0.0001,discount_factor=0.9,Slambda=0.0

# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 201 -x 200 -a alfa=0.0001,discount_factor=0.9,epsilon=0.4,Slambda=0.1 -q

class FeatSARSAAgent(Agent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=1000, discount_factor=0.9, Slambda=0.1, epsilon=0.1):

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

    def init_weights(self):

        self.weights = np.zeros(NFEATURES+1)
        self.ztrace  = np.zeros(NFEATURES+1)
        self.Q = self.weights

    def saveTable(self, filename):
        print("savetable qlearning")
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
            reward += 10

        if state.isWin():
            reward += 5000

        elif state.isLose():
            reward += -500

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
    def __init__ (self, alfa=0.01, maxa=1000, discount_factor=0.9, epsilon=0.1):

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

        self.last_feat = None

        #main weights
        self.weights  = None

        self.gamma_num_range = gamma_range
        self.e_num_range    = e_range
        self.training_number_range = training_range
        self.alfa_num_range = alfa_range
    

    def final(self, state):
        if state.isLose():
            self.reward = -500
        else:
            self.reward = 5000
        
        best_a, currQ = self.greedyAction(state)
        self.updateWeights(state, terminal=True)

        #print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epsilon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1

    def init_weights(self):
        self.weights = np.zeros(NFEATURES+1)
        #self.weights  = np.random.randn(NFEATURES+1)

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
