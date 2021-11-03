from game import Agent
from game import Directions
import util
import json
import numpy as np

grid_mapping = None
grid_width = 0

def get_state_id(state):
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
                cell_val = int(food_list[i][j]) + 2*int( (i,j) in ghosts_list) + int(2**2 * ((i,j) == pacman_position)) + (2**3 * int( (i,j) in scared_list)) + (2**4 * int( (i,j) in capsule_list))
                uuid += "_"+str(cell_val)
    return uuid

def get_position_by_id(state_id):
        cells = state_id.split("_")
        index = 0
        for i,c in enumerate(cells):
            if c != "":
                if int(c) & 4 > 0:
                    index = i-1
        x,y = grid_mapping[index]

        return (x,y)

class MCAgent(Agent):

    def set_grid_mapping(self, grid_width, grid_height, walls):
        c = 0
        global grid_mapping
        grid_mapping = dict()
        for i in range(grid_width):
            for j in range(grid_height):
                if(not walls[i][j]):
                    grid_mapping[c] = (i,j)
                    c+=1

    def update_heatmap(self, state_id, action):
        x,y = get_position_by_id(state_id)
        x = x-1 #To account for walls in the grid limit
        y = y-1 #To account for walls in the grid limit
        if self.state_action_rewards[state_id][action][0] > self.max_Q_val[x][y] \
                or self.max_Q_val[x][y] == 0 or self.max_Q_state[x][y] == None \
                or (self.max_Q_state[x][y][0] == state_id and self.max_Q_state[x][y][1] == action):
            self.max_Q_state[x][y] = ["",""]
            self.max_Q_state[x][y][0] = state_id
            self.max_Q_state[x][y][1] = action
            self.max_Q_val[x][y] = self.state_action_rewards[state_id][action][0]

    def __init__(self):
        #I use the same variable to count the number of times each state_action has been selected
        #self.state_action_rewards[state_id][action] is a list with [0] = reward, and [1] = count
        self.state_action_rewards = dict()
        self.output_path = None
        self.max_Q_state = None
        self.max_Q_val = None
        self.episode_size = 0
        self.episode = []

        #index is always 0 for pacman agents
        self.index = 0

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
                self.update_heatmap(state_id, selected)
        del visited_state_action

    def egreedy_policy(self, state, policy):
        rand = np.random.rand()
        state_id = get_state_id(state)
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
        state_id = get_state_id(state)
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
