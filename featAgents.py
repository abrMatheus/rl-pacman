from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util
from feat_utils import NFEATURES, computeDistances, filterDist


# python pacman.py -l customMaze -p FeatSARSAAgent -g PatrolGhost -n 101 -x 100 -a alfa=0.01,discount_factor=0.9,Slambda=0.0


class FeatSARSAAgent(Agent):
    "Function approximation SARSA"
    def __init__ (self, alfa=0.01, maxa=5000, discount_factor=0.9, Slambda=0.1, debug=False):

        #self.actions = ['West', 'East','Stop', 'North', 'South']
        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST) #, Directions.STOP)
        

        self.discount_factor = float(discount_factor)
        self.alfa = float(alfa)
        self.lambd = float(Slambda)

        self.i_epslon = 1 if self.alfa > 0 else 0.0
        self.epslon   = self.i_epslon

        self.last_w = None
        self.last_Q = None
        self.last_state = None
        self.is_train = True
        self.weights  = None
        self.total_reward = 0
        self.num_actions = 0
        self.ngames = 0

        self.time = 0

        ##########
        self.debug = bool(debug)

    def init_weights(self):

        self.weights = np.zeros(NFEATURES+1)
        self.ztrace  = np.zeros(NFEATURES+1)


    def final(self, state):
        if state.isLose():
            self.reward = -500
        else:
            self.reward = 5000
        
        best_a, currQ = self.greedyAction(state)
        self.updateWeights(state, Qvalue=currQ, terminal=True)

        print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward, "e", self.epslon
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1

    def getAction(self, state):

        if self.weights is None:
            self.init_weights()
    
        if not self.is_train:
            self.best_a, currQ = self.greedyAction(state)

        elif self.last_Q is None:
            #fist action
            self.best_a = self.randomAction(state)
            num_food = state.getNumFood()
            dists = computeDistances(state, num_food, self.directions)
            dists = filterDist(dists)
            currQ       = self.computeScore(self.best_a, dists)

        else:
            self.best_a, currQ = self.egreedyAction(state)
            self.reward = self.getReward(state)
            self.updateWeights(state, currQ)


        self.last_a = self.best_a
        self.last_Q = currQ
        self.last_state = state
        self.num_actions += 1
        self.time +=1
        return self.best_a
    

    def updateWeights(self, state,  Qvalue, terminal=False):


        td_error = self.reward

        if (self.debug):
            print("pacman pos", state.getPacmanPosition(), "ghost pos", state.getGhostPositions())
    
        dist = computeDistances(state, state.getNumFood(), self.directions)
        dist = filterDist(dist)
        currentFeat = self.featArray(self.best_a, dist)


        difference = (self.reward + self.discount_factor *Qvalue ) - self.last_Q
        if self.debug:
            print("last Q", self.last_Q, "Qvalue", Qvalue, "diff", difference)
            print("feat", currentFeat)
        #print("difference", difference)
        if self.debug:
            print("weights old", self.weights)
        self.weights = self.weights + self.alfa*difference*currentFeat 
        #self.weights = self.weights/(np.abs(self.weights).max()+0.1)
        if self.debug:
            print("weights", self.weights)
        self.total_reward +=self.reward



        # for i in range(NFEATURES+1):
        #     td_error -= self.last_weights[i]
        #     self.ztrace[i] += 1

        # if terminal:
        #     self.weights += self.alfa * td_error * self.ztrace
        #     #go to next episode
        # else :
        #     for i in range(NFEATURES+1) 
        #         td_error += td_error + self.lamb * self.weights[i]
            
        #     self.weights += self.alfa * td_error * self.ztrace
        #     self.ztrace  = self.ztrace*self.discount_factor*self.lambd



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
            score = self.computeScore(action, dists)
            scored.append((score, action))

        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]

        return random.choice(bestActions), bestScore

    def egreedyAction(self, state):
        p = random.random()

        if p > self.epslon:
            #print("greedy")
            action, score =  self.greedyAction(state)
            
        else:
            #print("random")
            action = random.choice(state.getLegalActions())

            successor = state.generateSuccessor(0, action)
            num_food = successor.getNumFood()
            dists = computeDistances(successor, num_food, self.directions)
            dists = filterDist(dists)
            score = self.computeScore(action, dists)


        self.epslon = self.i_epslon/(.2*self.ngames+1)
        #self.epslon  = self.i_epslon/np.log(self.time+0.1)
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

    def computeScore(self,action, dists):
        feature = self.featArray(action, dists)
        score = (feature*self.weights).sum()
        return score

    def featArray(self, action, dist):

        features = np.zeros([NFEATURES +1])

        features[0] = dist['food']                                        #1-distance to fodd
        if (dist['ghost']<3): features[1] = 1.0                           #2-have a ghost nearby
        #features[2] = 1/(dist['ghost']+0.001)                            #3-inverse distance to g
        features[2]  = dist['ghost']                                      #3-distance to ghost
        if (dist['ghost']<3 and dist['food']<2) : features [3] = 1.0      #4-ghost and food nearby
    
        features[-1] = 1                                                  #bias
        
        features = features/np.abs(features).sum()                        #normalize features 
        return features
