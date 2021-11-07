from game import Agent, Grid
from pacman import Actions, Directions

NFEATURES  = 4
MAX_DIST = 99999



def computeDistances(state, cur_num_foods, DIRECTIONS):


    # print("compute distance begin!!!")
    # print("************************")

    start = state.getPacmanPosition()
    # print("start", start, "ghost", state.getGhostPositions())
    walls = state.getWalls()
    seen = Grid(walls.width, walls.height, False)

    

    ghosts = getGhosts(state)
    ghost_dist = MAX_DIST

    # for a in range(walls.width):
    #     for b in range(walls.height):
    #         print("ghosts[",a,b,"]",ghosts[a][b])

    food_dist = MAX_DIST

    if state.getNumFood() < cur_num_foods:
        food_dist = 0

    Q = [(start[0], start[1], 0)]

    if ghosts[start[0]][start[1]] != 0:
        ghost_dist = 0

    while len(Q):
        curx, cury, curdist = Q.pop(0)
        
        for direction in DIRECTIONS:
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

            Q.append((x, y, dist))

    food_cg = getFoodCG(state)
    food_cg_dist = abs(food_cg[0] - start[0]) + abs(food_cg[1] - start[1])


    dist = {
        'food': food_dist, 
        'food_cg': food_cg_dist, 
        'ghost': ghost_dist, 
        'is_ghost': 1.0 if ghost_dist <= 1.75 else 0.0,
    }

    # print("end", dist)
    # print("*******************************************")
    return dist

def filterDist(dist):
    newdist = dist.copy()
    for (key, value) in dist.items():
        if value >= MAX_DIST:
            newdist[key] = -0.1
    return newdist

def getGhosts(state):
    ghosts = [
        s.getPosition() for s in state.getGhostStates()
        if s.scaredTimer <= 0
    ]
    return getMap(state, ghosts)

    return getMap(state, state.getGhostPositions())

def getMap(state, positions):
    layout = state.data.layout
    grid = Grid(layout.width, layout.height, 0)
    for i, (x, y) in enumerate(positions):
        grid[int(round(x))][int(round(y))] = i + 1
    return grid

def getFoodCG(state):
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

def normalizeDist(dist, maximum):
    return float(dist) / float(maximum) if dist < MAX_DIST else 1.0
