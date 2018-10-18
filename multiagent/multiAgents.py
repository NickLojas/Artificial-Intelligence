# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        Score, maxScore, minScore = successorGameState.getScore(), 999999, -999999
        
        if successorGameState.isWin():  return maxScore
        if successorGameState.isLose():  return minScore
        
        closestFood = min([util.manhattanDistance(newPos, food) for food in newFood.asList()])
        if closestFood == 0:
            closestFood = minScore
        
        ghostPosList = [ghost.getPosition() for ghost in newGhostStates]
        
        closestGhost = min([util.manhattanDistance(newPos, ghost) for ghost in ghostPosList])

        Score += closestGhost/(closestFood*2)
        return Score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        def MaxValue(gameState, depth):
            value = -999999
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                next_pos = gameState.generateSuccessor(0, action)
                value = max(value, MinValue(next_pos, depth, 1))
                
            return value
        
        def MinValue(gameState, depth, numGhost):
            value = 999999
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            for action in gameState.getLegalActions(numGhost):
                next_pos = gameState.generateSuccessor(numGhost, action)
                if gameState.getNumAgents() == numGhost + 1:
                    value = min(value, MaxValue(next_pos, depth + 1))
                else:
                    value = min(value, MinValue(next_pos, depth, numGhost + 1))
                
            return value
        
        #finding best action from the DFminimax
        bestAction, value = "", -999999
        legalActions = gameState.getLegalActions()
        for action in legalActions:
            next_pos = gameState.generateSuccessor(0, action)
            last_value = value
            value = max(value, MinValue(next_pos, 0, 1))
            if value > last_value: bestAction = action
            
        return bestAction
    
    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def MaxValue(gameState, depth, alpha, beta):
            value = -999999
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                next_pos = gameState.generateSuccessor(0, action)
                value = max(value, MinValue(next_pos, depth, 1, alpha, beta))
                if value >= beta: return value
                alpha = max(alpha, value)
                
            return value
        
        def MinValue(gameState, depth, numGhost, alpha, beta):
            value = 999999
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            for action in gameState.getLegalActions(numGhost):
                next_pos = gameState.generateSuccessor(numGhost, action)
                if gameState.getNumAgents() == numGhost + 1:
                    value = min(value, MaxValue(next_pos, depth + 1, alpha, beta))
                    if value <= alpha: return value
                    beta = min(beta, value)
                else:
                    value = min(value, MinValue(next_pos, depth, numGhost + 1, alpha, beta))
                    if value <= alpha: return value
                    beta = min(beta, value)
                
            return value
        
        #finding best action from the DFminimax
        bestAction, value, alpha, beta = "", -999999, -999999, 999999
        legalActions = gameState.getLegalActions()
        for action in legalActions:
            next_pos = gameState.generateSuccessor(0, action)
            last_value = value
            value = max(value, MinValue(next_pos, 0, 1, alpha, beta))
            if value > last_value: bestAction = action
            if value >= beta: return bestAction
            alpha = max (value, alpha)
            
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def MaxValue(gameState, depth):
            value = -999999
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            pacmanActions = gameState.getLegalActions(0)
            for action in pacmanActions:
                next_pos = gameState.generateSuccessor(0, action)
                value = max(value, Chance(next_pos, depth, 1))
                
            return value

        def Chance(gameState, depth, numGhost):
            value = 0
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return self.evaluationFunction(gameState)
            
            totalActions = len(gameState.getLegalActions(numGhost))
            for action in gameState.getLegalActions(numGhost):
                next_pos = gameState.generateSuccessor(numGhost, action)
                if gameState.getNumAgents() == numGhost + 1:
                    value += MaxValue(next_pos, depth + 1)
                else:
                    value += Chance(next_pos, depth, numGhost + 1)
                    
            value = float(value / totalActions)
            return value
        
        
        bestAction, value = "", -999999
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            next_pos = gameState.generateSuccessor(0, action)
            prev = value
            value = max(value, Chance(next_pos, 0, 1))
            if value > prev: bestAction = action
            
        return bestAction
        
   
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 1. returned max score and min score for wins and losses respectively.
      2. The further away the closest ghost is, the higher the score will be.
      3. The closer the closest food is, the less score will be subtracted from the total (rewards getting close food)
      4. If near a capsule, rewards getting it since eating a ghost will give more points
      5. The less food is on the game, the less the score will be subtracted through newFoodLeft.\
      (this helps with pacman beind stuck on getting a single food when the next foods are far away)
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin(): return 999999
    if currentGameState.isLose(): return -999999
    
    
    totalScore = scoreEvaluationFunction(currentGameState)
    numGhosts = currentGameState.getNumAgents() - 1
    newFoodList = currentGameState.getFood().asList()
    newFoodLeft = len(newFoodList)
    capsulesLeft = len(currentGameState.getCapsules())
    
    #calculate closest ghostDist score
    distFromPacman = 1000
    for i in range(1,numGhosts+1):
        dist = util.manhattanDistance(currentGameState.getPacmanPosition(),\
                                      currentGameState.getGhostPosition(i))
        if dist < distFromPacman: distFromPacman = dist
    #distFromPacman = max(distFromPacman, 5)
    totalScore += distFromPacman
    
    #calculate closest foodDist score
    foodDist = 1000
    for food in newFoodList:
        dist = util.manhattanDistance(currentGameState.getPacmanPosition(), food)
        if dist < foodDist: foodDist = dist
    totalScore -= foodDist * 1.3
    
    #calculate capsules remaining
    totalScore -= capsulesLeft * 3
    
    #calculate food remaining
    totalScore -= newFoodLeft * 8
    
    return totalScore
    
    

# Abbreviation
better = betterEvaluationFunction

