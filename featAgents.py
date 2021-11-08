from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util
from feat_utils import NFEATURES, computeDistances, filterDist


# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 101 -x 100 -a alfa=0.0001,discount_factor=0.9,Slambda=0.0


class FeatSARSAAgent(Agent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=5000, discount_factor=0.9, Slambda=0.1, epsilon=0.1):

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

        self.last_feat = None

        #main weights
        self.weights  = None

    def init_weights(self):

        self.weights = np.zeros(NFEATURES+1)
        self.ztrace  = np.zeros(NFEATURES+1)


    def final(self, state):
        if state.isLose():
            self.reward = -500
        else:
            self.reward = 5000
        
        best_a, currQ = self.greedyAction(state)
        self.updateWeights(state, terminal=True)

        print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epsilon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1

    def getAction(self, state):

        if self.weights is None:
            self.init_weights()
    
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

        dists = computeDistances(state, state.getNumFood(), self.directions)
        dists = filterDist(dists)
        self.curr_feat = self.featArray(dists)
        currQ = self.computeScoreFromFeat(self.curr_feat)
        
        td_error = self.reward + self.discount_factor*currQ - oldQ

        #self.weights +=  self.alfa*( (td_error - oldQ)*curr_feat)
        self.weights += self.alfa * td_error * self.curr_feat

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


#python pacman.py -l customMaze -p FeatQLAgent -g PatrolGhost -n 201 -x 200 -a alfa=0.0001,discount_factor=0.9,epsilon=0.3 -q
class FeatQLAgent(FeatSARSAAgent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=5000, discount_factor=0.9, epsilon=0.1):

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

        self.last_feat = None

        #main weights
        self.weights  = None
    

    def init_weights(self):
        self.weights = np.zeros(NFEATURES+1)
        #self.weights  = np.random.randn(NFEATURES+1)

    def updateWeights(self, state, terminal=False):

        oldQ = self.computeScoreFromFeat(self.last_feat)

        dists = computeDistances(state, state.getNumFood(), self.directions)
        dists = filterDist(dists)
        self.curr_feat = self.featArray(dists)
        #currQ = self.computeScoreFromFeat(self.curr_feat)

        ###get the best score (or currQ)
        _, currQ = self.greedyAction(state)
        
        td_error = self.reward + self.discount_factor*currQ - oldQ

        #self.weights +=  self.alfa*( (td_error - oldQ)*curr_feat)
        self.weights += self.alfa * td_error * self.curr_feat

        self.total_reward += self.reward
