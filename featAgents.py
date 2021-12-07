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

    def featArray(self, dist,x,y):

        features = np.zeros([NFEATURES +1])

        features[0] = x
        features[1] = y
        #features[0] = dist['food']                                        #1-distance to fodd
        #if (dist['ghost']<3): features[1] = 1.0                           #2-have a ghost nearby
        #features[2] = 1/(dist['ghost']+0.001)                            #3-inverse distance to g
        #features[2]  = dist['ghost']                                      #3-distance to ghost
        #if (dist['ghost']<3 and dist['food']<2) : features [3] = 1.0      #4-ghost and food nearby
    
        features[-1] = 1                                                  #bias
        
        #features = features/np.abs(features).sum()                        #normalize features 
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
    def __init__ (self, alfa=0.01, maxa=1000, discount_factor=0.9, epsilon=0.1, log_name=None):

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
        print("score", state.getScore(), 'nactions', self.num_actions)
        self.num_actions = 0
        self.total_reward = 0
        self.reward=0
        self.ngames += 1

    def init_weights(self):
        print("INIT WEIGHTS")
        self.weights = np.zeros(NFEATURES+1)
        #self.weights  = np.random.randn(NFEATURES+1)
        self.Q = self.weights

    def updateWeights(self, state, terminal=False):

        print("curr feat", self.curr_feat)

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
#obs_size = env.observation_space.shape[0] 
#n_actions = env.action_space.n  
#HIDDEN_SIZE = 256

import torch
obs_size = NFEATURES+1
n_actions = 4
HIDDEN_SIZE = 256
# model = torch.nn.Sequential(
#              torch.nn.Linear(2, HIDDEN_SIZE),
#              torch.nn.ReLU(),
#              torch.nn.Linear(HIDDEN_SIZE, n_actions),
#              torch.nn.Softmax(dim=0)
#      )


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


from torch import nn

def _initialize_weights(amodel):
        for module in amodel.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print("if module", module)
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                #nn.init.normal_(module.weight, std=0.1)
                #nn.init.constant_(module.weight,0.)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                print("else module", module)
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        #exit()

def initialize_weights_random(amodel):
    # Initializes weights according to the DCGAN paper
    for m in amodel.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# _initialize_weights(model)


class Net(nn.Module):
    def __init__(self,xsize=28,ysize=28):
        super(Net, self).__init__()
        self.bn0   = nn.BatchNorm2d(3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn3   = nn.BatchNorm2d(32)
        #self.fc1 = nn.Linear(32 * (xsize-6) * (ysize-6), 256)
        self.fc1 = nn.Linear(32*xsize*ysize, 128)
        #self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 4)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.bn0(x)
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        #x = self.bn1(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        #x = self.bn2(x)
        x = F.relu(self.conv3(x))
        #x = self.pool(x)
        #x = self.bn3(x)
        print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x


#from torchsummary import summary
import torch.nn.functional as F

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


import matplotlib.pyplot as plt
#python pacman.py -l customMaze -p REAgent -g PatrolGhost -n 201 -x 200 -a alfa=0.0001,discount_factor=0.9,epsilon=0.3 -q
class REAgent(FeatSARSAAgent):
    "Function approximation SARSA"
    def __init__ (self, lr=0.003, horizon=500, maxa = 5000, gamma=0.99, log_name=None):
        self.lr = float(lr)
        self.horizon = int(horizon)
        self.maxa = int(maxa)
        self.gamma = float(gamma)
        self.num_actions = 0

        self.log_name = log_name

        self.initial=True
        self.is_train = True
        
        self.last_state = None


        self.actions = np.array(['North', 'East', 'South', 'West'])#, 'Stop']
        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST) #, Directions.STOP)

        self.transitions = []
        
        self.ngames = 0
        self.t = 1


        # self.model = model

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #print(self.model)

        self.loss_log = []
        self.episode_loss = 0
        self.total_reward = 0
        self.reward_log = []
        self.actions_log = []


        self.stack_feats = None


        #TODO remove
        self.islose = None


        # checkpoint = torch.load('model.pt')
        # model.load_state_dict(checkpoint['model_state_dict'])

        self.model = Net(5,5)
        #self.model = Net(12,10)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #initialize_weights_random(self.model)

        #_initialize_weights(self.model)


        self.current_map = None



    def final(self, state):
        if self.is_train:
            if state.isWin():
                print("win")
                self.reward = 10
            else:
                print("lose")
                self.reward = -10
            
            self.total_reward +=self.reward
            
            self.transitions.append([self.current_map.detach().cpu(), np.where(self.actions==self.chosed_action), self.reward])
            #TODO? adiciono o estado final?
            floss = self.update_model()
            
            print('episode R', self.total_reward, "ac", self.num_actions, "el", self.episode_loss)

            if self.log_name is not None:
               self.update_log(state)


            self.loss_log.append(self.episode_loss)
            self.actions_log.append(self.num_actions)
            self.reward_log.append(self.episode_loss)

            self.episode_loss = 0
            self.total_reward = 0 

            fig, axs = plt.subplots(3,1)
            axs[0].plot(self.loss_log)
            axs[0].set_ylabel('loss')

            axs[1].plot(self.actions_log,'.')
            axs[1].plot(moving_average(self.actions_log, 15))
            axs[1].set_ylabel('n_actions')

            axs[2].plot(self.reward_log,'.')
            axs[2].plot(moving_average(self.reward_log, 15))
            axs[2].set_ylabel('reward')

            plt.savefig('loss.png')

            plt.close()

        self.num_actions = 0
        self.t=1
        self.last_state=None

        # print("self.transitions", self.transitions)

    def getAction(self, state):

        #print("is training", self.is_train)
        
        if self.last_state is not None:
            #self.reward = self.getReward(state)
            self.transitions.append([self.current_map.detach().cpu(), np.where(self.actions==self.chosed_action), self.reward])
            self.total_reward +=self.reward


        if self.t == self.horizon and self.is_train:
            _ = self.update_model()
            self.t=1

        # GET ACTION
        
        self.curr_feat = torch.Tensor(state.getPacmanPosition())
        with torch.no_grad():
            self.model.eval()
            #act_prob = self.model(self.curr_feat)


            current_map = util.get_state_image(state, None).transpose(0,3,1,2)
            
            # plt.imshow(current_map[0].transpose(1,2,0))
            # plt.savefig("state_images/state2.png")
            
            self.current_map = torch.from_numpy(current_map).float().cuda()/255.

            # plt.imshow(self.current_map[0].cpu().permute(1,2,0))
            # plt.savefig("state_images/state3.png")

            act_prob = self.model(self.current_map)[0].detach().cpu().numpy()
    
        legal_actions = state.getLegalActions()
        # filtprob = np.isin(self.actions,legal_actions)*act_prob.data.numpy()
        # filtprob = filtprob/filtprob.sum()
        # action = np.random.choice(self.actions, p=filtprob)
        # self.best_a = action

        if self.is_train:
            self.chosed_action = np.random.choice(self.actions, p=act_prob)
        else:
            self.chosed_action = self.actions[np.argmax(act_prob)]
            

        if not self.chosed_action in legal_actions:
            self.best_a = 'Stop'
            self.reward = -2
        else:
            self.best_a = self.chosed_action
            self.reward = -1

        print("input", self.curr_feat,"actprob",act_prob, "a:", self.chosed_action, "best", self.best_a)

        if self.last_state is not None:
            if state.getNumFood() > self.last_state.getNumFood():
                self.reward = 10


        self.last_state = state
        self.num_actions += 1
        self.t +=1
        self.last_feat = self.curr_feat



        if self.num_actions >= self.maxa: #max number of actions
            state.data._lose = True
            self.islose = True

        return self.best_a

    def update_model(self):
        print("hora de atualizar")
        batch_Gvals = []
        self.model.train()
        for i in range(len(self.transitions)):
            new_Gval = 0
            power=0
            for j in range(i, len(self.transitions)):
                new_Gval = new_Gval + (
                            (self.gamma**power)*self.transitions[j][2])
                power +=1
            batch_Gvals.append(new_Gval)
        expected_returns_batch=torch.FloatTensor(batch_Gvals).cuda()
        #expected_returns_batch/=expected_returns_batch.max()        
        #expected_returns_batch/=torch.abs(expected_returns_batch).max()

        print("expect returns", expected_returns_batch)
        print("rewards", [r for (s,a,r) in self.transitions])

        feat_batch = torch.Tensor([s.squeeze().numpy() for (s,a,r) in self.transitions]).cuda()
        action_batch = torch.Tensor([a for (s,a,r) in self.transitions]).cuda()


        pred_batch = self.model(feat_batch)
        #pred_batch = self.model(feat_batch)
        prob_batch = pred_batch.gather(dim=1,index=action_batch
                 .long().view(-1,1)).squeeze()

        print("prob_batch", torch.log(prob_batch) * expected_returns_batch)

        loss = -torch.sum(torch.log(prob_batch-0.05) * expected_returns_batch)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_loss += loss.item()

        self.transitions = []
        self.t = 1
        
        print("loss ", loss.item())
        return loss.item()

    def update_log(self, state):
            self.rlog.append(self.total_reward)
            self.slog.append(state.getScore())

            tmp = {
                'score' : self.slog,
                'reward': self.rlog
            }

            with open(self.log_name, "wb") as handle:
                pickle.dump(tmp, handle)


    def getReward(self, state):

        reward = -1
        #reward = 0 

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 10

        if state.isWin():
            reward += 5000

        elif state.isLose():
            reward += -500

        if self.best_a == 'Stop':
            reward += 0

        return reward
    


    # def saveTable(self, filename):
    #     tdict = {'weights' : self.weights}

    #     with open(filename, 'wb') as handle:
    #         pickle.dump(tdict, handle)

    
    # #----------------------------------------------------------------------------
    # #Functions used for the grid search algorithm
    # def get_parameters_in_iteration(self, i):
    #     e_size  = self.e_num_range.shape[0]
    #     a_size   = self.alfa_num_range.shape[0]
    #     g_size   = self.gamma_num_range.shape[0]
    #     ntr_size = self.training_number_range.shape[0]

    #     ntr = i%ntr_size
    #     g   = (i//ntr_size)%g_size
    #     a   = i//(g_size*ntr_size)%a_size
    #     e  = i//(a_size*g_size*ntr_size)%e_size

    #     return self.e_num_range[e], self.alfa_num_range[a], self.gamma_num_range[g], self.training_number_range[ntr]

    # def get_param_names(self):
    #     return ['Epsilon', 'Alpha', 'Gamma', 'N']

    # def set_parameters_in_iteration(self, i):
    #     e, a, g, ntr = self.get_parameters_in_iteration(i)
    #     self.epsilon = e
    #     self.alfa = a
    #     self.discount_factor = g

    #     return ntr

    # def get_training_space_size(self):
    #     e  = self.e_num_range.shape[0]
    #     a   = self.alfa_num_range.shape[0]
    #     g   = self.gamma_num_range.shape[0]
    #     ntr = self.training_number_range.shape[0]
        

    #     return e*a*g*ntr

    # def reset_tables(self):
    #     self.weights[:] = 0
    #     #self.episode_size = 0
    #     #self.episode = []
    #     self.reward=0
    #     self.last_state = None
    #     self.last_feat = None
    #     self.num_actions = 0
    #     self.total_reward = 0

    # def write_best_parameters(self, best_parameters, average_score):
    #     best_e = best_parameters[0]
    #     best_alfa = best_parameters[1]
    #     best_gamma = best_parameters[2]
    #     best_n_training = best_parameters[3]
    #     print("E : ", best_e)
    #     print("Alfa : ", best_alfa)
    #     print("Gamma : ", best_gamma)
    #     print("N_training : ", best_n_training)
    #     print("Average Score : ", average_score)

    #----------------------------------------------------------------------------


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        value = F.relu(self.critic_linear1(x))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(x))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist

#class A2CAgent(FeatSARSAAgent):
class A2CAgent(Agent):
    "A2C"
    def __init__ (self, lr=0.003, horizon=500, maxa = 5000, gamma=0.99, log_name=None):


        
        self.maxa = int(maxa)
        self.lr = float(lr)
        self.gamma = float(gamma)

        self.reset_values()

        self.model = ActorCritic(num_inputs=2, num_actions=4, hidden_size=256)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)


        self.actions = np.array(['North', 'East', 'South', 'West'])#, 'Stop']

        print("init!")


        self.toplot_rewards = []
        self.toplot_numact  = []
        self.toplot_loss    = []

    def reset_values(self):
        self.num_actions = 0
        self.last_state = None

        self.rewards   = []
        self.log_probs = []
        self.values    = []
        self.entropy_term = 0
        self.total_reward = 0



    def final(self, state):

        self.reward = self.getReward(state)

        pos = torch.Tensor(state.getPacmanPosition()).unsqueeze(0)
        qval, _ = self.model.forward(pos)

        loss = self.update_model(qval.detach().numpy()[0][0])

        win = 'LOSE'
        if state.isWin():
            win = 'WIN'

        #print("rewards", len(self.rewards), "values", len(self.values), "logs", len(self.log_probs))
        print("Episode num_a", self.num_actions, "R:", self.total_reward, "L", loss, win)


        if self.num_actions % 10 == 0:
            fig, axs = plt.subplots(3,1)
            axs[0].plot(self.toplot_loss)
            axs[0].plot(moving_average(self.toplot_loss, 15))
            axs[0].set_ylabel('loss')

            axs[1].plot(self.toplot_numact,'.')
            axs[1].plot(moving_average(self.toplot_numact, 15))
            axs[1].set_ylabel('n_actions')

            axs[2].plot(self.toplot_rewards,'.')
            axs[2].plot(moving_average(self.toplot_rewards, 15))
            axs[2].set_ylabel('reward')

            plt.savefig('loss.png')

            plt.close()
        

        self.reset_values()


    def update_model(self, i_qval):
        print("rewards", self.rewards)

        Qvals = np.zeros_like(self.values, dtype=np.float32)
        qval = i_qval
        for t in reversed(range(len(self.rewards))):
            qval = self.rewards[t] + self.gamma * qval
            Qvals[t] = qval
        
        Qvals  = torch.from_numpy(Qvals)
        values = torch.FloatTensor(self.values)
        #TODO what?
        log_probs = torch.stack(self.log_probs, dim=2)
        

        advantage = Qvals - values
        print("dims", log_probs.shape, advantage.shape)
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        a_loss = actor_loss + critic_loss + 0.001 * self.entropy_term
        
        ac_loss = a_loss

        self.optim.zero_grad()
        ac_loss.backward()
        self.optim.step()

        self.toplot_loss.append(ac_loss.item())
        self.toplot_rewards.append(self.total_reward)
        self.toplot_numact.append(self.num_actions)

        return ac_loss.item()

    
    def getAction(self, state):
        
        # get reward
        if self.last_state is not None:
            #not first action
            self.reward = self.getReward(state)




        # get action based on module
        action = self.getcriticAction(state)
        
            
        self.num_actions +=1
        self.last_state = state

        if self.num_actions >= self.maxa:
            state.data._lose = True
        

        return action


    def getReward(self,state):
        retvalue = -1
        if self.invalid_action:
            retvalue = -2
        
        elif state.isWin():
            retvalue = 10
        elif state.isLose():
            retvalue = -10
        
        elif state.getNumFood() < self.last_state.getNumFood():
            retvalue = 1
        else:
            retvalue = -1


        self.rewards.append(retvalue)
        self.values.append(self.i_value)
        self.log_probs.append(self.i_prob_action)
        dist = self.i_prob_action.detach().numpy()
        self.entropy_term += -np.sum(np.mean(dist) * np.log(dist))
        self.total_reward += retvalue


        return retvalue
    
    def getcriticAction(self, state):

        self.invalid_action = False

        # self.model.eval()
        # with torch.no_grad():
        pos = torch.Tensor(state.getPacmanPosition()).unsqueeze(0)
        self.i_value, self.i_prob_action  = self.model.forward(pos)
        chosed_action = np.random.choice(self.actions, p=self.i_prob_action.cpu().detach().numpy().squeeze(0))

        legal_actions = state.getLegalActions()

        retaction = chosed_action

        if not chosed_action in legal_actions:
            self.invalid_action = True
            retaction = 'Stop'

        print("p", pos[0], "prob", self.i_prob_action.cpu().detach()[0], chosed_action, retaction)

        return retaction

