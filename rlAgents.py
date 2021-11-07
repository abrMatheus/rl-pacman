from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random


MAX_DIST = 99999
NFEATURES = 3


class AQLAgent(Agent):
    "Q Learning Agent"

    def __init__ (self, path, learning_rate=None, discount_factor=None):
        self.actions = ['West', 'East','Stop', 'North', 'South']
        self.stateCode = {'West' : [-1, 0], 'East': [1, 0],'Stop': [0, 0], 'North': [0, 1], 'South': [0, -1]}

        if learning_rate is None or discount_factor is None:
            raise ValueError('Learning rate and Discount Factor must be provided')

        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST)
        self.last_a = None
        self.currentFeat = None
        self.discount_factor = float(discount_factor)
        self.LR = float(learning_rate)
        self.num_actions = 0
        self.total_reward = 0
        self.i_epslon =  1.0 if self.LR > 0 else 0.0
        self.epslon = 0.9
        try:
            with open('rl-data/run.pickle', 'rb') as handle:
                loaddata = pickle.load(handle)

            self.weights = loaddata['weights']
        except:
            self.weights = np.zeros(NFEATURES+1)
        

    def saveWeights (self):
        tdict = {'weights': self.weights}
        with open('rl-data/run.pickle', 'wb') as handle:
            pickle.dump(tdict, handle)

    def final(self, state):
        if state.isLose() :
            self.reward += -500
        else:
            self.reward += 5000
        best_a, currQ = self.greedyAction(state)
        self.updateWeights(currQ)

        print '# Actions:', self.num_actions,'# Total Reward:', self.total_reward
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0

        self.saveWeights()

    def updateWeights(self, maxQs):
        #print("reward", self.reward)
        difference = (self.reward + self.discount_factor *maxQs ) - self.currentQ
        #print("difference", difference)
        self.weights = self.weights + self.LR*difference*self.currentFeat 
        self.weights = self.weights/np.abs(self.weights.sum()+0.1)
        #print("weights", self.weights)
        
        self.total_reward +=self.reward

    def getAction(self, state):
        best_a, currQ = self.egreedyAction(state)

        #verifica se eh a primeira acao:
        if self.currentFeat is not None:
            maxQs = currQ
            self.updateWeights(maxQs)

        pacpos = np.array(state.getPacmanPosition())
        posactions = state.getLegalPacmanActions()

        #---------- feature -----------

        dist = self.computeDistances(state, state.getNumFood(), len(state.getCapsules()))
        dist = self.filterDist(dist)
        self.currentFeat = self.featArray(best_a, dist)
        #-----------------------------

        #predict the next state and reward
        next_s = pacpos + self.stateCode[best_a]
        self.reward = self.predictReward(best_a, state, next_s)

        self.currentQ =  (self.currentFeat*self.weights).sum()

        self.last_a = best_a
        self.num_actions += 1
        return best_a


    def predictReward(self, best_a, state, next_s):
        currentFood = state.getFood()
        ghosts = self.getGhosts(state)
        capsules = self.getCapsules(state)
        edible_ghosts = self.getEdibleGhosts(state)
        num_food = state.getNumFood()

        x,y = next_s
        reward = 0

        if( currentFood[x][y]):
            reward += 10
        
        return reward

    


    def getFoodList(self, currentFood):
        M = currentFood.height
        N = currentFood.width

        foods =[]
        for y in range(M):
            for x in range(N):
                if currentFood[x][y] == True:
                    foods.append([x,y])
        
        return np.array(foods)

    def featArray(self, action, dist):
        taken_action_index = self.actions.index(action)

        features = np.zeros([NFEATURES +1])

        features[0] = dist['food']                                              #1-ok
        if (dist['ghost']<3): features[1] = 1.0                                 #2-um so fantasma
        features[2] = dist['ghost']                                             #3-ok
        if (dist['ghost']<3 and dist['food']<2) : features [1] = 1.0
        
        
        features[-1] = 1                                                        #bias
        #print("features", features)
        features = features/np.abs(features).sum() #normalize features 
        return features

    def computeDistances(self, state, cur_num_foods, cur_num_capsules):

        start = state.getPacmanPosition()
        walls = state.getWalls()
        seen = Grid(walls.width, walls.height, False)

        ghosts = self.getGhosts(state)
        ghost_dist = MAX_DIST

        edible_ghosts = self.getEdibleGhosts(state)
        edible_ghost_dist = MAX_DIST
        is_edible = False
        #is_edible = state.getGhostState(1).scaredTimer > 0 #TODO:verify this

        capsules = self.getCapsules(state)
        caps_dist = MAX_DIST

        food_dist = MAX_DIST

        if state.getNumFood() < cur_num_foods:
            food_dist = 0

        if len(state.getCapsules()) < cur_num_capsules:
            caps_dist = 0

        Q = [(start[0], start[1], 0)]

        while len(Q):
            if (ghost_dist != MAX_DIST and caps_dist != MAX_DIST and food_dist != MAX_DIST and
                (not is_edible or edible_ghost_dist != MAX_DIST)):
                break
            curx, cury, curdist = Q.pop(0)

            for direction in self.directions:
                dx, dy = Actions.directionToVector(direction)
                x, y = int(curx + dx), int(cury + dy)
                dist = curdist + 1

                if state.hasWall(x, y) or seen[x][y]:
                    continue

                seen[x][y] = True
                if state.hasFood(x, y):
                    food_dist = min(dist, food_dist)

                if ghosts[x][y] != 0:
                    ghost_dist = min(dist, ghost_dist)

                if edible_ghosts[x][y] != 0:
                    edible_ghost_dist = min(dist, edible_ghost_dist)

                if capsules[x][y] != 0:
                    caps_dist = min(dist, caps_dist)

                Q.append((x, y, dist))

        food_cg = self.getFoodCG(state)
        food_cg_dist = abs(food_cg[0] - start[0]) + abs(food_cg[1] - start[1])


        dist = {
            'food': food_dist, 
            'food_cg': food_cg_dist, 
            'ghost': ghost_dist, 
            'edible_ghost': edible_ghost_dist,
            'capsules': caps_dist,
            'is_ghost': 1.0 if ghost_dist <= 1.75 else 0.0,
        }
        return dist

    def filterDist(self, dist):
        newdist = dist.copy()
        for (key, value) in dist.items():
            if value >= MAX_DIST:
                newdist[key] = -0.1
        return newdist

    def maskToPosition(self, mask):
        positions = []
        for x in range(mask.width):
            for y in range(mask.height):
                if mask[x][y]:
                    positions.append((x, y))
        return positions

    def getGhosts(self, state):
        ghosts = [
            s.getPosition() for s in state.getGhostStates()
            if s.scaredTimer <= 0
        ]
        return self.getMap(state, ghosts)

        return self.getMap(state, state.getGhostPositions())

    def getEdibleGhosts(self, state):
        ghosts = [
            s.getPosition() for s in state.getGhostStates()
            if s.scaredTimer > 0
        ]
        return self.getMap(state, ghosts)

    def getCapsules(self, state):
        return self.getMap(state, state.getCapsules())

    def getMap(self, state, positions):
        layout = state.data.layout
        grid = Grid(layout.width, layout.height, 0)
        for i, (x, y) in enumerate(positions):
            grid[int(round(x))][int(round(y))] = i + 1
        return grid

    def getFoodCG(self, state):
        food = state.getFood()
        count = 0
        x, y = 0, 0
        for i in range(food.width):
            for j in range(food.height):
                if food[i][j]:
                    count += 1
                    x += i
                    y += j
        if count == 0:
            return MAX_DIST, MAX_DIST
        return x / count, y / count

    def normalizeDist(self, dist, maximum):
        return float(dist) / float(maximum) if dist < MAX_DIST else 1.0

    def computeScore(self,action, dists):
        feature = self.featArray(action, dists)
        score = (feature*self.weights).sum()
        return score

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
            dists = self.computeDistances(state, num_food, num_capsules)
            dists = self.filterDist(dists)
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
            num_capsules = len(successor.getCapsules())
            dists = self.computeDistances(successor, num_food, num_capsules)
            dists = self.filterDist(dists)
            score = self.computeScore(action, dists)


        self.epslon = self.i_epslon/(0.05*self.num_actions+1)
        return action, score