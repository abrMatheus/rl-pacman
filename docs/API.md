# States API


The main method of an Agent is the *getAction(self, state)*, which will update their direction based on the *state* parameter. This section aims to describe the state API.


## getCapsules
  Returns a list of positions (x,y) of the remaining capsules. For example: 

  ```
  [(2, 3)]
  ```

## getFood
Returns a Grid of boolean food indicator variables.

Grids can be accessed via list notation, so to check
if there is food at (x,y), just call

    currentFood = state.getFood()
    if currentFood[x][y] == True

or for the entire map (testClassic)

    FFFFFFFFF
    FFFFFTTTF
    FFFFFFFFF
    FFFFFFFFF
    FTFFTFFTF
    FTFFTTTTF
    FFFFFFFFF

## getGhostPositions
Returns an list of postitions (x,y) of each ghost.For example:
    
    [(2, 5)]


## getLegalActions
Returns an list with of possible actions. For example:

    ['Stop', 'East', 'South']


## getNumFood

Returns the number of remaining foods


## getPacmanPosition

Returns the (x,y) pacman position


## getScore
Returns a float with the score


## getWalls

Returns a Grid of boolean wall indicator variables.

Grids can be accessed via list notation, so to check
if there is a wall at (x,y), just call

    walls = state.getWalls()
    if walls[x][y] == True

or the entire wall (testClassic):


    TTTTTTTTT
    TFFFTFFFT
    TTTTTTTFT
    TFFFFFFFT
    TFTTFTTFT
    TFTTFFFFT
    TTTTTTTTT

## isLose

Returns a boolean

## isWin
Returns a boolean
