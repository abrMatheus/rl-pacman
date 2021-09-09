from game import Agent
from game import Directions


class DumbAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."

        print "Location: ", state.getPacmanPosition()
        print "Actions available: ", state.getLegalPacmanActions()

        if Directions.WEST in state.getLegalPacmanActions():
            print "Going West."
            return Directions.WEST
        else:
            print "Going West."
            return Directions.STOP

