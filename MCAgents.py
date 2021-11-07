from game import Agent
from game import Directions
from pacman import Actions
import util
import json
import numpy as np
from game import Grid

class MCAgent(Agent):
    #CLASS CONSTRUCTOR
    def __init__(self, gamma = 0.9, epsilon_num = 10.0):
        #I use the same variable to count the number of times each state_action has been selected
        #self.Q[state_id][action] is a list with [0] = reward, and [1] = count
        self.Q = dict()
        self.episode_size = 0
        self.episode = []

        #index is always 0 for pacman agents
        self.index = 0

        self.gamma_range = np.array([0.9])
        self.epsilon_num_range = np.array([1.0, 10.0, 100.0, 1000.0])
        self.training_number_range = np.array([1000,10000,50000])
        self.last_state = None
        self.is_train = False

        #Monte-carlo parameters
        self.gamma = gamma
        self.epsilon_num = epsilon_num

    #----------------------------------------------------------------------------
    #Functions used for the Monte Carlo Control algorithm
    def getDistribution( self, state ):
        dist = util.Counter()
        possible_actions = state.getLegalActions( self.index )
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
                if state_id not in self.Q:
                    self.Q[state_id] = dict()
                if selected not in self.Q[state_id]:
                    self.Q[state_id][selected] = [0,0]
                self.Q[state_id][selected][1] += 1
                alpha = (1.0/(self.Q[state_id][selected][1]))
                Qsa = self.Q[state_id][selected][0]
                self.Q[state_id][selected][0] = Qsa + alpha * (G - Qsa)
                if(util.max_Q_val is not None):
                    util.update_heatmap(state_id, selected, self.Q[state_id][selected][0])
        del visited_state_action

    def egreedy_policy(self, state, policy):
        rand = np.random.rand()
        state_id = util.get_state_id(state)
        state_count = 0
        if state_id in self.Q:
            for a_ in self.Q[state_id]:
                state_count+=self.Q[state_id][a_][1]
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
        action = self.egreedy_policy(state, self.Q)

        #new_state = state.generateSuccessor(self.index, action)

        #self.episode.append([state_id, action, new_state.getScore()])
        self.episode.append([state_id, action, 0])
        #print(self.episode)
        if self.episode_size > 0:
            self.episode[self.episode_size-1][2] = self.episode[self.episode_size-2][2] + self.getReward(state)

        if state_id not in self.Q:
            self.Q[state_id] = dict()
        if action not in self.Q[state_id]:
            self.Q[state_id][action] = [0,0]

        if(self.episode_size > 5000):
            state.data._lose = True

        #if(new_state.isWin() or new_state.isLose()):

        self.episode_size+=1
        self.last_state = state
        return action

    def final(self, state):
        self.episode[self.episode_size-1][2] = state.getScore()
        if(state.data._lose):
            self.episode[self.episode_size-1][2] = self.episode[self.episode_size-2][2] - 500
        if(self.is_train):
            self.updateQ()
        self.episode_size = 0
        del self.episode
        self.episode = []

    #----------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    #Function used to save the state_action_reward table as a json file
    def saveTable(self, output_model_path):
        file_out = open(output_model_path, "w")

        json.dump(self.Q, file_out)

        file_out.close()
    #----------------------------------------------------------------------------


    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
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
        del self.Q
        self.Q = dict()
        self.episode_size = 0
        self.episode = []

    def getReward(self, state):
        reward = -1

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 10
        if state.isWin():
            reward += 5000
        elif state.isLose():
            reward += -500

        return reward

    def write_best_parameters(self, best_parameters, average_score):
        best_epsilon = best_parameters[0]
        best_gamma = best_parameters[1]
        best_n_training = best_parameters[2]
        print("Epsilon : ", best_epsilon)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    def get_param_names(self):
        return ['Epsilon', 'Gamma', 'N']
    #----------------------------------------------------------------------------


class MCFunctionAgent(Agent):
    #CLASS CONSTRUCTOR
    def __init__(self, gamma = 0.9, epsilon_num = 10.0, alpha = 0.01, total_episodes=10):
        self.episode_size = 0
        self.episode = []

        #index is always 0 for pacman agents
        self.index = 0
        self.previous_xsa = 0
        self.num_features = 7
        self.weights = np.zeros([self.num_features + 1])

        self.gamma_range = np.array([0.9])
        self.epsilon_num_range = np.array([1.0, 10.0, 100.0, 1000.0])
        self.training_number_range = np.array([1000,10000,50000])
        self.last_state = None
        self.is_train = False
        self.elapsed_episodes = 0
        self.total_episodes = total_episodes
        self.max_dist = 99999999
        self.directions = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST)
        self.actions = ['West', 'East', 'North', 'South']

        self.previous_action = None

        self.eps_a = .5
        self.eps_b = .1
        self.eps_c = .1

        self.alpha = alpha

        self.distances = []

        #Monte-carlo parameters
        self.gamma = gamma
        self.epsilon_num = epsilon_num

    #----------------------------------------------------------------------------
    #Functions used for the Monte Carlo Control algorithm
    def getDistribution( self, state ):
        dist = util.Counter()
        possible_actions = state.getLegalActions( self.index )
        for a in possible_actions: dist[a] = 1.0
        dist.normalize()
        return dist

    def getStateActionFeatures(self, state, action):

        dist = self.computeDistances(state)
        for (key, value) in dist.items():
            if value >= self.max_dist:
                dist[key] = -0.1

        features = np.zeros([self.num_features + 1])

        features[0] = dist['food']

        if dist['ghost'] < 2:
            features[1] = 1.0

        features[2] = dist['ghost']

        features[3] = 1 / dist['ghost']

        features[4] = dist['edible_ghost']

        #if dist['edible_ghost'] <= 2:
        #    features[5] = 1.0

        #features[6] = dist['capsule']

        #if dist['ghost'] < 3 and dist['capsule'] < 2:
        #    features[7] = 1.0

        if dist['ghost'] > 3 and dist['food'] < 3:
            features[5] = 1.0

        if self.previous_action == action:
            features[6] = 1.0

        features[7] = 1

        features = features / np.abs(features).sum()

        return features

    def getStateActionFeatures(self, state, action):

        dist = self.computeDistances(state)
        for (key, value) in dist.items():
            if value >= self.max_dist:
                dist[key] = -0.1

        features = np.zeros([self.num_features + 1])

        features[0] = dist['food']

        if dist['ghost'] < 2:
            features[1] = 1.0

        features[2] = dist['ghost']

        features[3] = 1 / dist['ghost']

        features[4] = dist['edible_ghost']

        #if dist['edible_ghost'] <= 2:
        #    features[5] = 1.0

        #features[6] = dist['capsule']

        #if dist['ghost'] < 3 and dist['capsule'] < 2:
        #    features[7] = 1.0

        if dist['ghost'] > 3 and dist['food'] < 3:
            features[5] = 1.0

        if self.previous_action == action:
            features[6] = 1.0

        features[7] = 1

        features = features / np.abs(features).sum()

        return features

    def getGhostsGridPositions(self, state):
        ghosts = [ s.getPosition() for s in state.getGhostStates() if s.scaredTimer <= 0 ]

        return self.getGridPositions(state, ghosts)

    def getEdibleGhostsGridPositions(self, state):
        edible_ghosts = [ s.getPosition() for s in state.getGhostStates() if s.scaredTimer > 0 ]

        return self.getGridPositions(state, edible_ghosts)

    def getGridPositions(self, state, positions):
        layout = state.data.layout

        grid = Grid(layout.width, layout.height, 0)

        for i, (x, y) in enumerate(positions):
            grid[int(round(x))][int(round(y))] = i + 1

        return grid

    def getFoodCentroid(self, state):
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
            return self.max_dist, self.max_dist
        return x / count, y / count

    def computeDistances(self, state):

        start = state.getPacmanPosition()
        walls = state.getWalls()
        seen = Grid(walls.width, walls.height, False)

        ghost_grid_position = self.getGhostsGridPositions(state)
        ghost_dist = self.max_dist

        edible_ghost_grid_position = self.getEdibleGhostsGridPositions(state)
        edible_ghost_dist = self.max_dist

        is_edible = False

        caps_grid_position = self.getCapsulesGridPositions(state)
        caps_dist = self.max_dist

        food_dist = self.max_dist

        if self.previous_num_food is not None and state.getNumFood() < self.previous_num_food:
            food_dist = 0

        if self.previous_num_capsule is not None and len(state.getCapsules()) < self.previous_num_capsule:
            caps_dist = 0

        Q = [(start[0], start[1], 0)]

        while len(Q):

            if (ghost_dist != self.max_dist and caps_dist != self.max_dist and food_dist != self.max_dist and
                (not is_edible or edible_ghost_dist != self.max_dist)):
                break

            current_x, current_y, current_dist = Q.pop(0)

            for direction in self.directions:

                dx, dy = Actions.directionToVector(direction)

                x, y = int(current_x + dx), int(current_y + dy)

                dist = current_dist + 1

                if state.hasWall(x, y) or seen[x][y]:
                    continue

                seen[x][y] = True
                if state.hasFood(x, y):
                    food_dist = min(dist, food_dist)

                if ghost_grid_position[x][y] != 0:
                    ghost_dist = min(dist, ghost_dist)

                if edible_ghost_grid_position[x][y] != 0:
                    edible_ghost_dist = min(dist, edible_ghost_dist)

                if caps_grid_position[x][y] != 0:
                    caps_dist = min(dist, caps_dist)

                Q.append((x, y, dist))

        food_centroid = self.getFoodCentroid(state)
        food_centroid_dist = abs(food_centroid[0] - start[0]) + abs(food_centroid[1] - start[1])

        dist = {
            'food': food_dist,
            'food_cg': food_centroid_dist,
            'ghost': ghost_dist,
            'edible_ghost': edible_ghost_dist,
            'capsule': caps_dist,
            'is_ghost': 1.0 if ghost_dist <= 1.75 else 0.0,
        }

        return dist

    def egreedyPolicy(self, state, legal_actions):

        if util.flipCoin(self.getEpsilon(self.elapsed_episodes)):
            # random action
            action = np.random.choice(legal_actions)
        else:
            # greedy action
            action = self.greedyPolicy(state)

        return action

    def greedyPolicy(self, state):
        # getting state_id
        legal_actions = state.getLegalActions()

        action_values = dict()
        for a in legal_actions:
            action_values[a] = 0.0

        for action in action_values.keys():
            xsa = self.getStateActionFeatures(state, action)
            q_saw = np.dot(xsa, self.weights)
            action_values[action] = q_saw

        action = max(action_values, key=action_values.get)

        return action

    def getAction(self, state):
        state_id = util.get_state_id(state)
        legal_actions = state.getLegalActions()
        action = self.egreedyPolicy(state, legal_actions)

        #new_state = state.generateSuccessor(self.index, action)

        #self.episode.append([state_id, action, new_state.getScore()])
        self.episode.append([state, action, 0])
        self.distances.append(self.computeDistances(state))
        #print(self.episode)
        if self.episode_size > 0:
            self.episode[self.episode_size-1][2] = self.episode[self.episode_size-2][2]  + self.getReward(state)

        if(self.episode_size > 5000):
            state.data._lose = True

        #if(new_state.isWin() or new_state.isLose()):

        self.episode_size+=1
        self.last_state = state
        return action

    def final(self, state):
        self.episode[self.episode_size-1][2] = self.episode[self.episode_size-2][2]  + self.getReward(state)
        if(state.data._lose):
            self.episode[self.episode_size-1][2] = self.episode[self.episode_size-2][2] -500
        if(self.is_train):
            self.updateWeights()
        self.elapsed_episodes+=1
        self.episode_size = 0
        del self.episode
        self.episode = []
        self.distances = []

    #----------------------------------------------------------------------------


    #---------------------------------------------------------------------------
    #Function used to save the state_action_reward table as a json file
    def saveTable(self, output_model_path):
        #file_out = open(output_model_path, "w")

        #json.dump(self.Q, file_out)

        #file_out.close()
        print("IMPLEMENT THE SAVE TABLE FUNCTION")
    #----------------------------------------------------------------------------


    #----------------------------------------------------------------------------
    #Functions used for the grid search algorithm
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
        del self.weights
        self.weights = np.zeros([self.num_features + 1])
        self.episode_size = 0
        self.episode = []
        self.distances = []

    def getReward(self, state):
        reward = -1

        if state.getNumFood() < self.last_state.getNumFood():
            reward += 10
        if state.isWin():
            reward += 5000
        elif state.isLose():
            reward += -500

        return reward

    def write_best_parameters(self, best_parameters, average_score):
        best_epsilon = best_parameters[0]
        best_gamma = best_parameters[1]
        best_n_training = best_parameters[2]
        print("Epsilon : ", best_epsilon)
        print("Gamma : ", best_gamma)
        print("N_training : ", best_n_training)
        print("Average Score : ", average_score)

    def get_param_names(self):
        return ['Epsilon', 'Gamma', 'N']
    #----------------------------------------------------------------------------


    def getCapsulesGridPositions(self, state):

        return self.getGridPositions(state, state.getCapsules())

    def computeDistances(self, state):
        start = state.getPacmanPosition()
        walls = state.getWalls()
        seen = Grid(walls.width, walls.height, False)

        ghost_grid_position = self.getGhostsGridPositions(state)
        ghost_dist = self.max_dist

        edible_ghost_grid_position = self.getEdibleGhostsGridPositions(state)
        edible_ghost_dist = self.max_dist

        is_edible = False

        caps_grid_position = self.getCapsulesGridPositions(state)
        caps_dist = self.max_dist

        food_dist = self.max_dist

        food_eaten = state.data.scoreChange > 1

        if food_eaten:
            food_dist = 0

        if False:
            caps_dist = 0

        Q = [(start[0], start[1], 0)]

        while len(Q):

            if (ghost_dist != self.max_dist and caps_dist != self.max_dist and food_dist != self.max_dist and
                (not is_edible or edible_ghost_dist != self.max_dist)):
                break

            current_x, current_y, current_dist = Q.pop(0)

            for direction in self.directions:

                dx, dy = Actions.directionToVector(direction)

                x, y = int(current_x + dx), int(current_y + dy)

                dist = current_dist + 1

                if state.hasWall(x, y) or seen[x][y]:
                    continue

                seen[x][y] = True
                if state.hasFood(x, y):
                    food_dist = min(dist, food_dist)

                if ghost_grid_position[x][y] != 0:
                    ghost_dist = min(dist, ghost_dist)

                if edible_ghost_grid_position[x][y] != 0:
                    edible_ghost_dist = min(dist, edible_ghost_dist)

                if caps_grid_position[x][y] != 0:
                    caps_dist = min(dist, caps_dist)

                Q.append((x, y, dist))

        food_centroid = self.getFoodCentroid(state)
        food_centroid_dist = abs(food_centroid[0] - start[0]) + abs(food_centroid[1] - start[1])

        dist = {
            'food': food_dist,
            'food_cg': food_centroid_dist,
            'ghost': ghost_dist,
            'edible_ghost': edible_ghost_dist,
            'capsule': caps_dist,
            'is_ghost': 1.0 if ghost_dist <= 1.75 else 0.0,
        }

        return dist

    def getEpsilon(self, elapsed_episodes):
        std_elapsed_episodes = (elapsed_episodes - self.eps_a * self.total_episodes) \
                            / (self.eps_b * self.total_episodes)
        cosh = np.cosh(np.exp(-std_elapsed_episodes))
        epsilon = 1.1 - (1 / cosh + (elapsed_episodes * self.eps_c / self.total_episodes))

        return epsilon

    def updateWeights(self):
            # getting reward
            episode = self.episode

            self.previous_action = "Stop"

            for i in range(len(episode)):
                reward = episode[i][2]
                action = episode[i][1]
                state = episode[i][0]
                xsa = self.getStateActionFeatures(state, action)

                q_saw_prime = np.dot(xsa, self.weights)

                q_saw = np.dot(self.previous_xsa, self.weights)

                self.weights += self.alpha * (reward + self.gamma * q_saw_prime - q_saw) * self.previous_xsa
                self.previous_xsa = xsa
                self.previous_action = action
