from game import Agent, Grid
from pacman import Actions, Directions
import numpy as np
import pickle
import os
import random
import util


gamma_range    = np.array([0.9])
alfa_range     = np.array([0.1, 0.2])
training_range = np.array([500, 2000, 4000])
lambda_range   = np.array([0.1, 0.2, 0.5, 0.7])
n0_range       = np.array([0.1, 10, 100])

class SARSAAgent(Agent):
    "Sarsa Lambda agent"

    def __init__ (self, n0 =0.6, maxa=5000, discount_factor=0.9, Slambda=0.7, log_name=None):
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

        self.gamma_num_range = gamma_range
        self.n0_num_range    = n0_range
        self.lambda_num_range = lambda_range
        self.training_number_range = training_range

        self.initQ = None
        self.qstucked = False

        self.log_name = log_name
        self.rlog = []
        self.slog = []

    def saveTable(self, filename):
        tdict = {'qtable'  : self.Q,
                 'statesL' : self.statesL}
        
        with open(filename, 'wb') as handle:
            pickle.dump(tdict, handle)

    
    def final(self, state):
        if self.is_train and not self.qstucked:
            self.reward = self.getReward(state)
            self.updateQ(state)

        if self.is_train:
            diffQ = np.abs(self.Q - self.initQ).sum()
            #print "diff Q:", diffQ, "total reward", self.total_reward

            if diffQ < 3 and not self.qstucked:
                print("Really stucked!!!", self.total_reward, self.num_actions)
                self.qstucked = True

            if self.log_name is not None:
                self.update_log(state)

            #print '# GAME:', self.Ngames,'# Actions:', self.num_actions,'# Total Reward:', self.total_reward
            self.Ngames +=1
            self.reward=0
            self.last_a=None
            self.last_state = None
            self.E_trace[:] = 0
            self.num_actions = 0
            self.total_reward = 0

    def update_log(self, state):
            self.rlog.append(self.total_reward)
            self.slog.append(state.getScore())

            tmp = {
                'score' : self.slog,
                'reward': self.rlog
            }

            with open(self.log_name, "wb") as handle:
                pickle.dump(tmp, handle)


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

        #nstates = n_positions*(2**num_ghosts)*(2**num_capsules)*(2**num_foods)
        nstates  = (n_positions**num_ghosts)*((2*num_foods)*(n_positions-num_foods)) + (2**(num_foods-1) * num_foods)

        self.Q = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.statesL = [util.get_state_id(state)]
        self.E_trace = np.zeros([nstates, len(self.actions)], dtype=np.float16)
        self.Ntable  = np.zeros([nstates, len(self.actions)], dtype=np.int64)

    def getAction(self, state):

        if len(self.Q) <=0 :
            self.init_states(state)

        self.num_actions +=1
        if self.num_actions > self.maxa or (self.qstucked and self.is_train): #max number of actions
            state.data._lose = True
            return 'Stop'

        if not self.is_train: 
            # teste
            sindex = self.getTableIndex(state)
            self.best_a = self.greedyAction(state,sindex)
        elif self.last_a is None: 
            #first state
            del self.initQ
            self.best_a = self.egreedyAction(state)
            self.initQ = self.Q.copy()
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


        lens = len(self.statesL)
        self.Q[:lens, :] +=alfa*td_error*self.E_trace[:lens, :]
        self.E_trace[:lens, :] = self.gamma * self.lambd * self.E_trace[:lens, :]

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



    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
    def get_parameters_in_iteration(self, i):
        n0_size  = self.n0_num_range.shape[0]
        l_size   = self.lambda_num_range.shape[0]
        g_size   = self.gamma_num_range.shape[0]
        ntr_size = self.training_number_range.shape[0]

        ntr = i%ntr_size
        g   = (i//ntr_size)%g_size
        l   = i//(g_size*ntr_size)%l_size
        n0  = i//(l_size*g_size*ntr_size)%n0_size

        return self.n0_num_range[n0], self.lambda_num_range[l], self.gamma_num_range[g], self.training_number_range[ntr]

    def get_param_names(self):
        return ['N0', 'Lambda', 'Gamma', 'N']

    def set_parameters_in_iteration(self, i):
        n0, l, g, ntr = self.get_parameters_in_iteration(i)
        self.N0 = n0
        self.lambd = l
        self.gamma = g

        return ntr

    def get_training_space_size(self):
        n0  = self.n0_num_range.shape[0]
        l   = self.lambda_num_range.shape[0]
        g   = self.gamma_num_range.shape[0]
        ntr = self.training_number_range.shape[0]
        

        return n0*l*g*ntr

    def reset_tables(self):
        self.Q[:] = 0
        self.E_trace[:] = 0
        self.Ntable[:]  = 0
        self.episode_size = 0
        self.episode = []
        self.statesL = []
        self.qstucked = False
        self.Ngames = 0 
        self.reward=0
        self.last_a=None
        self.last_state = None
        self.num_actions = 0
        self.total_reward = 0

    def write_best_parameters(self, best_parameters, average_score):
        best_n0 = best_parameters[0]
        best_lambda = best_parameters[1]
        best_gamma = best_parameters[2]
        best_n_training = best_parameters[3]
        print("N0 : ", best_n0)
        print("Lambda : ", best_lambda)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    #----------------------------------------------------------------------------

