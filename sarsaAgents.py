from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import time

class SARSAAgent(Agent):
    "Sarsa Lambda agent"

    def __init__ (self, path='rl-data/',n0 =100, maxa=5000, discount_factor=0.9, Slambda=0.9):
        self.actions = ['West', 'East','Stop', 'North', 'South']
        self.statesL = None

        self.gamma = float(discount_factor)
        self.N0     = float(n0)
        self.last_a = None
        self.lambd = float(Slambda)
        self.maxa  = float(maxa)

        self.Training = True  if self.N0 > 0 else False

        self.num_actions = 0
        self.total_reward = 0
        self.Ngames = 0

        self.filename = os.path.join(path, 'SARSAAgent.pickle')

        if os.path.isfile(self.filename):
            print "Loading Q_table ---------------------------------------"
            with open(self.filename, 'rb') as handle:
                loaddata = pickle.load(handle)
                self.Q_table = loaddata['qtable']
                self.statesL = loaddata['statesL']
                self.Vmatrix = loaddata['vmatrix']
                self.E_trace = np.zeros(self.Q_table.shape, dtype=np.float16)
                self.Ntable  = np.zeros(self.Q_table.shape, dtype=np.int64)

        else:
            self.Q_table = None

    def saveStates(self):
        tdict = {'qtable'  : self.Q_table,
                 'statesL' : self.statesL,
                 'vmatrix' : self.Vmatrix}
        
        with open(self.filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    
    def final(self, state):
        self.reward = self.getReward(state)
        self.updateQ(state)

        print '# GAME:', self.Ngames,'# Actions:', self.num_actions,'# Total Reward:', self.total_reward
        self.Ngames +=1
        self.reward=0
        self.saveStates()
        self.last_a=None
        self.last_state = None
        self.E_trace[:] = 0
        self.num_actions = 0
        self.total_reward = 0



    def getTableIndex(self, state):
        uuid = self.get_state_id(state)
        if not uuid in self.statesL:
            self.statesL.append(uuid)

        return self.statesL.index(uuid)
        

    def init_states(self, state):
        #compute the number of states
        num_foods    = state.getNumFood()
        num_capsules = len(state.getCapsules())
        num_ghosts   = len(state.getGhostPositions())
        n_positions  = 0
        walls = state.getWalls()

        x_size = walls.width
        y_size = walls.height
        for i in range(x_size):
            for j in range(y_size):
                if(not walls[i][j]):
                    n_positions += 1

        nstates = n_positions*(2**num_ghosts)*(2**num_capsules)*(2**num_foods)

        self.Q_table = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.statesL = [self.get_state_id(state)]
        self.E_trace = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.Ntable  = np.zeros([nstates, len(self.actions)], dtype=np.int64)
        self.Vmatrix = np.zeros([x_size, y_size])

    def getAction(self, state):

        if self.Q_table is None:
            self.init_states(state)

        self.num_actions +=1
        if self.num_actions > self.maxa: #max number of actions
            state.data._lose = True
            return 'Stop'

        if not self.Training: 
            # teste
            sindex = self.getTableIndex(state)
            self.best_a = self.greedyAction(state,sindex)
        elif self.last_a is None: 
            #first state
            self.best_a = self.egreedyAction(state)
        else:
            self.best_a = self.egreedyAction(state)
            self.reward = self.getReward(state)
            self.updateQ(state)
            
        self.last_a = self.best_a
        self.last_state = state
        return self.best_a


    def updateQ(self, state):
        sindex  = self.getTableIndex(self.last_state)
        aindex  = self.actions.index(self.last_a)
        s2index = self.getTableIndex(state)
        a2index = self.actions.index(self.best_a)

        priorq   = self.Q_table[sindex, aindex]
        td_error = self.reward + self.gamma * self.Q_table[s2index, a2index] - priorq

        self.E_trace[sindex, aindex] +=1

        self.Ntable[sindex, aindex] +=1

        alfa = 1./float(self.Ntable[sindex, aindex])
        self.Q_table +=alfa*td_error*self.E_trace
        self.E_trace = self.gamma * self.lambd * self.E_trace

        x,y = state.getPacmanPosition()
        currMax = self.Q_table[s2index, :].max()
        if (currMax > self.Vmatrix[x,y]):
            self.Vmatrix[x,y] = currMax

        self.total_reward += self.reward


    def getReward(self, state):

        reward = -1 

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 10

        if state.isWin():
            reward += 5000

        elif state.isLose():
            reward += -500

        return reward


    def greedyAction(self, state, sindex):
        avalues = self.Q_table[sindex,:]

        legal_actions = state.getLegalActions()
        

        scored = []
        for legal_a in legal_actions:

            tmpindex = self.actions.index(legal_a)

            score = avalues[tmpindex]

            scored.append((score, tmpindex))


        
        scored=np.array(scored)
        bestindex = np.argmax(scored[:,0])


        bestActions = self.actions[int(scored[bestindex,1])]
        bestScore   = scored[bestindex,0]
        

        return bestActions



    def egreedyAction(self, state):
        p = random.random()

        sindex = self.getTableIndex(state)
        n = float(self.Ntable[sindex, :].sum())

        curr_epslon = self.N0/(self.N0+n)

        if p > curr_epslon:
            action =  self.greedyAction(state,sindex)
        else:
            legal_actions = state.getLegalActions()
            action = random.choice(legal_actions)
  
        return action


    def get_state_id(self, state):
        ghosts_list = state.getGhostPositions()
        ghosts = state.getGhostStates()
        scared_list = []
        for g in ghosts:
            if g.scaredTimer > 0:
                position = g.getPosition()
                scared_list.append(position)
                ghosts_list.remove(position)
        food_list = state.getFood()
        pacman_position = state.getPacmanPosition()
        walls = state.getWalls()
        capsule_list = state.getCapsules()

        x_size = food_list.width
        y_size = food_list.height

        uuid = ""
        for i in range(x_size):
            for j in range(y_size):
                if(not walls[i][j]):

                    cell_val = int(food_list[i][j]) + 2*int( (i,j) in ghosts_list) + int(22 * ((i,j) == pacman_position)) + (23 * int( (i,j) in scared_list)) + (2**4 * int( (i,j) in capsule_list))
                    uuid += str(cell_val)
        return uuid








class NSARSAAgent(Agent):
    "Normal Sarsa Agent"

    def __init__ (self, path='rl-data/', learning_rate=0.001, maxa=5000, discount_factor=0.9):
        self.actions = ['West', 'East','Stop', 'North', 'South']
        self.statesL = None

        self.gamma = float(discount_factor)
        self.last_a = None
        self.maxa  = maxa
        self.alfa  = float(learning_rate)

        self.num_actions = 0
        self.total_reward = 0

        self.i_epslon =  1.0 if self.alfa > 0 else 0.0
        self.epslon = 0.9

        self.Training = True  if self.alfa > 0 else False

        self.filename = os.path.join(path, 'NSARSAAgent.pickle')
        self.Ngames = 0 

        if os.path.isfile(self.filename):
            print "Loading Q_table ---------------------------------------"
            with open(self.filename, 'rb') as handle:
                loaddata = pickle.load(handle)
                self.Q_table = loaddata['qtable']
                self.statesL = loaddata['statesL'] #TODO: nao preciso
                self.Vmatrix = loaddata['vmatrix']

        else:
            self.Q_table = None

    def saveStates(self):
        tdict = {'qtable'  : self.Q_table,
                 'statesL' : self.statesL,
                 'vmatrix' : self.Vmatrix}
        
        with open(self.filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    
    def final(self, state):
        self.reward = self.getReward(state)
        self.updateQ(state)

        print '# GAME:', self.Ngames,'# Actions:', self.num_actions,'# Total Reward:', self.total_reward
        self.Ngames +=1
        self.reward=0
        self.saveStates()
        self.last_a=None
        self.last_state = None
        self.num_actions = 0
        self.total_reward = 0



    def getTableIndex(self, state):
        uuid = self.get_state_id(state)
        if not uuid in self.statesL:
            self.statesL.append(uuid)

        return self.statesL.index(uuid)
        

    def init_states(self, state):
        #compute the number of states
        num_foods    = state.getNumFood()
        num_capsules = len(state.getCapsules())
        num_ghosts   = len(state.getGhostPositions())
        n_positions  = 0
        walls = state.getWalls()

        x_size = walls.width
        y_size = walls.height
        for i in range(x_size):
            for j in range(y_size):
                if(not walls[i][j]):
                    n_positions += 1

        nstates = n_positions*(2**num_ghosts)*(2**num_capsules)*(2**num_foods)

        self.Q_table = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.statesL = [self.get_state_id(state)]
        self.Vmatrix = np.zeros([x_size, y_size])

    def getAction(self, state):

        if self.Q_table is None:
            self.init_states(state)

        self.num_actions +=1
        if self.num_actions > self.maxa:
            state.data._lose = True
            return 'Stop'


        if not self.Training:
            sindex = self.getTableIndex(state)
            self.best_a = self.greedyAction(state,sindex)

        elif self.last_a is None:
            self.best_a = self.egreedyAction(state)

        else:

            self.best_a = self.egreedyAction(state)

            self.reward = self.getReward(state)

            self.updateQ(state)
            
        self.last_a = self.best_a
        self.last_state = state           
        return self.best_a


    def updateQ(self, state):
        sindex  = self.getTableIndex(self.last_state)
        aindex  = self.actions.index(self.last_a)
        s2index = self.getTableIndex(state)
        a2index = self.actions.index(self.best_a)

        self.total_reward += self.reward

        priorq   = self.Q_table[sindex, aindex]
        td_error = self.reward + self.gamma * self.Q_table[s2index, :].max() - priorq

        self.Q_table[sindex, aindex] += self.alfa*td_error
        x,y = state.getPacmanPosition()
        currMax = self.Q_table[s2index, :].max()
        if (currMax > self.Vmatrix[x,y]):
            self.Vmatrix[x,y] = currMax



    def getReward(self, state):

        reward = -1 

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 10

        if state.isWin():
            reward += 5000

        elif state.isLose():
            reward += -500


        return reward


    def greedyAction(self, state, sindex):
        
        avalues = self.Q_table[sindex,:]

        legal_actions = state.getLegalActions()

        scored = []
        for legal_a in legal_actions:

            tmpindex = self.actions.index(legal_a)

            score = avalues[tmpindex]

            scored.append((score, tmpindex))


        
        scored=np.array(scored)
        bestindex = np.argmax(scored[:,0])


        bestActions = self.actions[int(scored[bestindex,1])]
        bestScore   = scored[bestindex,0]
        

        return bestActions



    def egreedyAction(self, state):
        p = random.random()

        sindex = self.getTableIndex(state)
        if p > self.epslon:
            action =  self.greedyAction(state,sindex)
        else:
            legal_actions = state.getLegalActions()
            action = random.choice(legal_actions)

        self.epslon = self.i_epslon/np.exp(self.num_actions*0.25+0.001)
        return action


    def get_state_id(self, state):
        ghosts_list = state.getGhostPositions()
        ghosts = state.getGhostStates()
        scared_list = []
        for g in ghosts:
            if g.scaredTimer > 0:
                position = g.getPosition()
                scared_list.append(position)
                ghosts_list.remove(position)
        food_list = state.getFood()
        pacman_position = state.getPacmanPosition()
        walls = state.getWalls()
        capsule_list = state.getCapsules()

        x_size = food_list.width
        y_size = food_list.height

        uuid = ""
        for i in range(x_size):
            for j in range(y_size):
                if(not walls[i][j]):

                    cell_val = int(food_list[i][j]) + 2*int( (i,j) in ghosts_list) + int(22 * ((i,j) == pacman_position)) + (23 * int( (i,j) in scared_list)) + (2**4 * int( (i,j) in capsule_list))
                    uuid += str(cell_val)
        return uuid
