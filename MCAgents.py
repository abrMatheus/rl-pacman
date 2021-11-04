from game import Agent
from game import Directions
import util
import json
import numpy as np





class MCAgent(Agent):
    def __init__(self):
        #I use the same variable to count the number of times each state_action has been selected
        #self.state_action_rewards[state_id][action] is a list with [0] = reward, and [1] = count
        self.state_action_rewards = dict()
        self.output_path = None
        self.episode_size = 0
        self.episode = []

        #index is always 0 for pacman agents
        self.index = 0

        self.gamma_range = np.arange(0.1, 1.0, 0.4)
        self.gamma_range = np.concatenate([self.gamma_range, np.arange(0.95, 1.01, 0.05)])

        self.epsilon_num_range = np.arange(10,101,40)
        self.epsilon_num_range = np.concatenate([self.epsilon_num_range, np.arange(100,1001,400)])
        self.epsilon_num_range = np.concatenate([self.epsilon_num_range, np.arange(1000,10001,4000)])

        self.training_number_range = np.array([100,1000, 10000, 100000, 500000, 1000000])

        #Monte-carlo parameters
        self.gamma = 0.9
        self.epsilon_num = 100.0

    #Function used to save the state_action_reward table as a json file
    def saveTable(self):
        file_out = open(self.output_path, "w")

        json.dump(self.state_action_rewards, file_out)

        file_out.close()

    def getDistribution( self, state ):
        dist = util.Counter()
        possible_actions = state.getLegalActions( self.index )
        #if Directions.STOP in possible_actions:
        #    possible_actions.remove( Directions.STOP )
        #if len(possible_actions) > 1:
        #    if Directions.REVERSE[state.getPacmanState().configuration.direction] in possible_actions:
        #        possible_actions.remove(Directions.REVERSE[state.getPacmanState().configuration.direction])
        for a in possible_actions: dist[a] = 1.0
        dist.normalize()
        return dist

    def updateQ(self):
        G = 0
        visited_state_action = dict()
        c = 1
        for i in range(len(self.episode)-1, 0, -1):
            #c+=1
            state_id = self.episode[i][0]
            if(state_id not in visited_state_action):
                visited_state_action[state_id] = []
            selected = self.episode[i][1]
            reward = self.episode[i][2]
            G = (self.gamma **c * G) + reward
            if selected not in visited_state_action[state_id]:
                visited_state_action[state_id].append(selected)
                if state_id not in self.state_action_rewards:
                    self.state_action_rewards[state_id] = dict()
                if selected not in self.state_action_rewards[state_id]:
                    self.state_action_rewards[state_id][selected] = [0,0]
                self.state_action_rewards[state_id][selected][1] += 1
                alpha = (1.0/(self.state_action_rewards[state_id][selected][1]))
                Qsa = self.state_action_rewards[state_id][selected][0]
                self.state_action_rewards[state_id][selected][0] = Qsa + alpha * (G - Qsa)
                if(util.max_Q_val is not None):
                    util.update_heatmap(self, state_id, selected)
        del visited_state_action

    def egreedy_policy(self, state, policy):
        rand = np.random.rand()
        state_id = util.get_state_id(state)
        state_count = 0
        if state_id in self.state_action_rewards:
            for a_ in self.state_action_rewards[state_id]:
                state_count+=self.state_action_rewards[state_id][a_][1]
        epsilon = self.epsilon_num/(self.epsilon_num+state_count)
        possible_actions = state.getLegalActions(self.index)
        #possible_actions.remove(Directions.STOP)
        if(rand < epsilon):
            #RANDOM
            dist = util.Counter()
            for a in possible_actions:
                dist[a] = 1.0
            dist.normalize()
            if len(dist) == 0:
                selected = Directions.STOP
            else:
                selected = util.chooseFromDistribution( dist )
        else:
            #GREEDY
            if(state_id not in policy):
                policy[state_id] = dict()
                for a in possible_actions:
                    policy[state_id][a] = 0

            policy_ = policy[state_id].values()
            values = [item[0] for item in policy_]

            tied_indices = []
            best_val = max(values)
            for i,val in enumerate(values):
                if best_val == val:
                    tied_indices.append(i)
            tied_best_actions = []
            for indx in tied_indices:
                tied_best_actions.append(policy[state_id].keys()[indx])
            possible_best_actions = []
            for v in tied_best_actions:
                if v in possible_actions:
                    possible_best_actions.append(v)

            dist = util.Counter()
            for a in possible_best_actions: dist[a] = 1.0
            dist.normalize()

            if len(dist) == 0:
                return Directions.STOP
            else:
                selected = util.chooseFromDistribution( dist )
        return selected

    def getAction(self, state):
        state_id = util.get_state_id(state)
        action = self.egreedy_policy(state, self.state_action_rewards)

        #new_state = state.generateSuccessor(self.index, action)

        #self.episode.append([state_id, action, new_state.getScore()])
        self.episode.append([state_id, action, 0])
        #print(self.episode)
        if self.episode_size > 0:
            self.episode[self.episode_size-1][2] = state.getScore()

        if state_id not in self.state_action_rewards:
            self.state_action_rewards[state_id] = dict()
        if action not in self.state_action_rewards[state_id]:
            self.state_action_rewards[state_id][action] = [0,0]


        #if(self.episode_size > 5000):
        #    self.episode_size = 0
        #    del self.episode
        #    self.episode = []
        #    state.data._lose = True

        #if(new_state.isWin() or new_state.isLose()):

        self.episode_size+=1
        return action

    def final(self, state):
        self.episode[self.episode_size-1][2] = state.getScore()
        self.updateQ()
        self.episode_size = 0
        del self.episode
        self.episode = []

    def get_parameters_in_iteration(self, i):
        x_size = self.epsilon_num_range.shape[0]
        y_size = self.gamma_range.shape[0]
        z_size = self.training_number_range.shape[0]

        z = i%z_size
        y = (i//z_size)%y_size
        x = i//(y_size*z_size)%x_size

        return self.epsilon_num_range[x], self.gamma_range[y], self.training_number_range[z]

    def set_parameters_in_iteration(self, i):
        x, y, z = self.get_parameters_in_iteration(i)
        self.epsilon_num = x
        self.gamma = y

        return z

    def get_training_space_size(self):
        x = self.epsilon_num_range.shape[0]
        y = self.gamma_range.shape[0]
        z = self.training_number_range.shape[0]

        return x*y*z

    def reset_tables(self):
        del self.state_action_rewards
        self.state_action_rewards = dict()
        self.episode_size = 0
        self.episode = []

    def write_best_parameters(self, best_parameters, average_score, output_file_path):
        best_epsilon = best_parameters[0]
        best_gamma = best_parameters[1]
        best_n_training = best_parameters[2]
        print("Epsilon : ", best_epsilon)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

        output_file=open(output_file_path,mode="w")
        string_to_write = "Epsilon : " + str(best_epsilon) + "\n"
        output_file.write(string_to_write)
        string_to_write = "Gamma : " + str(best_gamma) + "\n"
        output_file.write(string_to_write)
        string_to_write = "N_training : " + str(best_n_training) + "\n"
        output_file.write(string_to_write)
        string_to_write = "Average Score : " + str(average_score) + "\n"
        output_file.write(string_to_write)
        output_file.close()
