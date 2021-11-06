from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util

class SARSAAgent(Agent):
    "Sarsa Lambda agent"

    def __init__ (self, n0 =0.6, maxa=5000, discount_factor=0.99, Slambda=0.7):
        self.actions = ['West', 'East','Stop', 'North', 'South']
        self.statesL = None

        self.gamma = float(discount_factor)
        self.N0     = float(n0)
        self.last_a = None
        self.lambd = float(Slambda)
        self.maxa  = float(maxa)

        self.is_train = True  if self.N0 > 0 else False

        self.num_actions = 0
        self.total_reward = 0
        self.Ngames = 0

        self.Q = []

        self.gamma_range = np.array([0.9])
        self.alfa_range = np.array([0.1, 0.2])
        self.training_number_range = np.array([200])

    def saveTable(self, filename):
        tdict = {'qtable'  : self.Q,
                 'statesL' : self.statesL}
        
        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    
    def final(self, state):
        if self.is_train:
            self.reward = self.getReward(state)
            self.updateQ(state)

        #print '# GAME:', self.Ngames,'# Actions:', self.num_actions,'# Total Reward:', self.total_reward
        self.Ngames +=1
        self.reward=0
        self.last_a=None
        self.last_state = None
        self.E_trace[:] = 0
        self.num_actions = 0
        self.total_reward = 0



    def getTableIndex(self, state):
        uuid = util.get_state_id(state)
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

        self.Q = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.statesL = [util.get_state_id(state)]
        self.E_trace = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.Ntable  = np.zeros([nstates, len(self.actions)], dtype=np.int64)

    def getAction(self, state):

        if len(self.Q) <=0 :
            self.init_states(state)

        self.num_actions +=1
        if self.num_actions > self.maxa: #max number of actions
            state.data._lose = True
            return 'Stop'

        if not self.is_train: 
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

        priorq   = self.Q[sindex, aindex]
        td_error = self.reward + self.gamma * self.Q[s2index, a2index] - priorq

        self.E_trace[sindex, aindex] +=1

        self.Ntable[sindex, aindex] +=1

        alfa = 1./float(self.Ntable[sindex, aindex])
        self.Q +=alfa*td_error*self.E_trace
        self.E_trace = self.gamma * self.lambd * self.E_trace


        self.total_reward += self.reward

        if(util.max_Q_val is not None):
            state_id = util.get_state_id(state)
            util.update_heatmap(state_id, self.last_a, self.Q[sindex][aindex])


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
        avalues = self.Q[sindex,:]

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



class NSARSAAgent(Agent):
    "Normal Sarsa Agent"

    def __init__ (self, learning_rate=0.2, maxa=5000, discount_factor=0.99):
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

        self.is_train = True  if self.alfa > 0 else False

        self.Ngames = 0 

        self.Q = []


        self.gamma_range          = np.array([0.9])
        self.alfa_num_array       = np.array([0.1, 0.2])
        self.training_number_range  = np.array([1000]) 

    def saveTable(self, filename):
        tdict = {'qtable'  : self.Q,
                 'statesL' : self.statesL}
        
        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    
    def final(self, state):
        
        if self.is_train:
            self.reward = self.getReward(state)
            self.updateQ(state)

        #print '# GAME:', self.Ngames,'# Actions:', self.num_actions,'# Total Reward:', self.total_reward
        self.Ngames +=1
        self.reward=0
        #self.saveStates()
        self.last_a=None
        self.last_state = None
        self.num_actions = 0
        self.total_reward = 0



    def getTableIndex(self, state):
        uuid = util.get_state_id(state)
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

        self.Q = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.statesL = [util.get_state_id(state)]

    def getAction(self, state):

        if len(self.Q) <= 0 :
            self.init_states(state)

        self.num_actions +=1
        if self.num_actions > self.maxa:
            state.data._lose = True
            return 'Stop'


        if not self.is_train:
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

        priorq   = self.Q[sindex, aindex]
        td_error = self.reward + self.gamma * self.Q[s2index, :].max() - priorq

        self.Q[sindex, aindex] += self.alfa*td_error

        if(util.max_Q_val is not None):
            state_id = util.get_state_id(state)
            util.update_heatmap(state_id, self.last_a, self.Q[sindex][aindex])



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
        
        avalues = self.Q[sindex,:]

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


    #----------------------------------------------------------------------------


    # self.gamma_range          = np.array([0.9])
    # self.alfa_num_array       = np.array([0.1, 0.2])
    # self.trainning_number_range  = np.array([1000]) 


    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):
        x_size = self.alfa_num_array.shape[0]
        y_size = self.gamma_range.shape[0]
        z_size = self.training_number_range.shape[0]

        z = i%z_size
        y = (i//z_size)%y_size
        x = i//(y_size*z_size)%x_size

        return self.alfa_num_array[x], self.gamma_range[y], self.training_number_range[z]

    def set_parameters_in_iteration(self, i):
        x, y, z = self.get_parameters_in_iteration(i)
        self.alfa = x
        self.gamma = y

        return z

    def get_training_space_size(self):
        x = self.alfa_num_array.shape[0]
        y = self.gamma_range.shape[0]
        z = self.training_number_range.shape[0]

        return x*y*z

    def reset_tables(self):
        self.Q[:] = 0
        self.episode_size = 0
        self.episode = []

    def write_best_parameters(self, best_parameters, average_score, output_file_path):
        best_alfa = best_parameters[0]
        best_gamma = best_parameters[1]
        best_n_training = best_parameters[2]
        print("Alfa : ", best_alfa)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

        output_file=open(output_file_path,mode="w")
        string_to_write = "Alfa : " + str(best_alfa) + "\n"
        output_file.write(string_to_write)
        string_to_write = "Gamma : " + str(best_gamma) + "\n"
        output_file.write(string_to_write)
        string_to_write = "N_training : " + str(best_n_training) + "\n"
        output_file.write(string_to_write)
        string_to_write = "Average Score : " + str(average_score) + "\n"
        output_file.write(string_to_write)
        output_file.close()

    #----------------------------------------------------------------------------