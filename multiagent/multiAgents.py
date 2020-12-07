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


    def getAction(self, state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a state and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = state.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(state, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentstate, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        states (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a state (pacman.py)
        successorstate = currentstate.generatePacmanSuccessor(action)
        newPos = successorstate.getPacmanPosition()
        newFood = successorstate.getFood()
        newGhostStates = successorstate.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        "*** YOUR CODE HERE ***"
        #take the new score and the current so we can calculate the dif
        newScore = successorstate.getScore()
        cur_score = scoreEvaluationFunction(currentstate)
        score = newScore - cur_score
        
        #find the closest ghost
        closer_ghost = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])

        #find closest food
        foodList = newFood.asList()
        if foodList:
            closeFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            closeFoodDist = 0

        #see if we can eat ghosts
        smallScareTime = min(newScaredTimes)
        if smallScareTime != 0:
            closer_ghost = 0
        
        
        ret_num = (10/(closeFoodDist+1)) #plus one so its not divided with zero
        ret_num = ret_num +(closer_ghost/10)
        ret_num = ret_num + score
        return ret_num


    
        #return successorstate.getScore()

def scoreEvaluationFunction(currentstate):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentstate.getScore()

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
   
    def getAction(self, state):
        """
        Returns the minimax action from the current state using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        state.getLegalActions(agent):
        Returns a list of legal actions for an agent
        agent=0 means Pacman, ghosts are >= 1

        state.generateSuccessor(agent, action):
        Returns the successor game state after an agent takes an action

        state.getNumAgents():
        Returns the total number of agents in the game

        state.isWin():
        Returns whether or not the game state is a winning state

        state.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #agent is always 0
        def max_funct(state,depth):
            #add one for next depth
            currDepth = depth + 1

            #cheack if we have to terminate 
            if state.isWin() or state.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(state)
            
            maxvalue = -100000
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0,action)
                #call min
                min_ret = min_funct(successor,currDepth,1)
                #find the max value
                maxvalue = max (maxvalue,min_ret)
            return maxvalue
        
        
        def min_funct(state,depth, agent):
            minvalue = 100000
            if state.isWin() or state.isLose():   #Terminal Test 
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agent)
            for action in actions:
                successor= state.generateSuccessor(agent,action)
                if agent == (state.getNumAgents() - 1):
                    #call max
                    max_ret = max_funct(successor,depth)
                    #find min value
                    minvalue = min (minvalue,max_ret)
                else:
                    #call min
                    min_ret = min_funct(successor,depth,agent+1)
                    #find min value
                    minvalue = min(minvalue,min_ret)

            return minvalue
        
        #call min the first time at the root
        
        cur_score = -999999
        actions = state.getLegalActions(0)
        for action in actions:
            nextState = state.generateSuccessor(0,action)
            score = min_funct(nextState,0,1)
            
            #keep the action with the max score
            if score > cur_score:
                ret_action = action
                cur_score = score
        return ret_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #agent is always 0
        def max_funct(state,depth,alpha,beta):
            
            #add one for next depth
            currDepth = depth + 1

            #cheack if we have to terminate 
            if state.isWin() or state.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(state)
            
            maxvalue = -100000
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0,action)
                #call min
                min_ret = min_funct(successor,currDepth,1,alpha,beta)
                #find the max value
                maxvalue = max (maxvalue,min_ret)
                if maxvalue > beta:
                    return maxvalue
                else:
                    alpha = max(alpha,maxvalue)
            return maxvalue
        
        
        def min_funct(state,depth, agent,alpha,beta):
            
            minvalue = 100000
            if state.isWin() or state.isLose():   #Terminal Test 
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agent)
            for action in actions:
                successor= state.generateSuccessor(agent,action)
                if agent == (state.getNumAgents() - 1):
                    #call max
                    max_ret = max_funct(successor,depth,alpha,beta)
                    #find min value
                    minvalue = min (minvalue,max_ret)
                    if(minvalue < alpha):
                        return minvalue
                    else:
                        beta = min(beta,minvalue)
                else:
                    #call min
                    min_ret = min_funct(successor,depth,agent+1,alpha,beta)
                    #find min value
                    minvalue = min(minvalue,min_ret)
                    if(minvalue < alpha):
                        return minvalue
                    else:
                        beta = min(beta,minvalue)

            return minvalue
        
        #call min the first time at the root
        
        cur_score = -100000
        alpha = -100000
        beta = 100000
        actions = state.getLegalActions(0)
        for action in actions:
            nextState = state.generateSuccessor(0,action)
            score = min_funct(nextState,0,1,alpha,beta)
            
            #keep the action with the max score
            if score > cur_score:
                ret_action = action
                cur_score = score
            
            #upbate alpha
            if score > beta:
                return ret_action
            else:
                alpha = max(alpha,score)
        return ret_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
    
        #agent is always 0
        def max_funct(state,depth):
            #add one for next depth
            currDepth = depth + 1

            #cheack if we have to terminate 
            if state.isWin() or state.isLose() or currDepth==self.depth:   
                return self.evaluationFunction(state)
            
            maxvalue = -100000
            actions = state.getLegalActions(0)
            for action in actions:
                successor= state.generateSuccessor(0,action)
                #call min
                min_ret = expValue(successor,currDepth,1)
                #find the max value
                maxvalue = max (maxvalue,min_ret)
            return maxvalue

        def expValue(state,depth ,agentIndex ):
            
            agents_count = state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)

            if state.isWin() or state.isLose():   
                return self.evaluationFunction(state)

            
            expect_value = 0
            probabilty = 1.0 / len(legalActions) #so we can multiply with exp_value
            #take all actions and call max or expect
            for action in legalActions:
                if agentIndex == agents_count - 1:
                    cur_expect_value =  max_funct(state.generateSuccessor(agentIndex, action),  depth)
                else:
                    cur_expect_value = expValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                expect_value += cur_expect_value * probabilty

            return expect_value
        
        #call first time
        actions = state.getLegalActions(0)
        cur_score = -100000
        for action in actions:
            nextState = state.generateSuccessor(0,action)
            # call expValue
            score = expValue(nextState,0,1)
            # take the best action
            if score > cur_score:
                ret_action = action
                cur_score = score
        return ret_action

        util.raiseNotDefined()

def betterEvaluationFunction(currentstate):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    I took the food distance and the closest ghost
    When the food dist is long the score gets smaller
    when the ghost is far the score gets bigger
    """
    
    
    
    "*** YOUR CODE HERE ***"
    if currentstate.isWin():
        return 10000
    
    #take info fron currentstate
    cur_pos = currentstate.getPacmanPosition()
    cur_food = currentstate.getFood().asList()
    cur_ghost_states = currentstate.getGhostStates()
    cur_scared_times = [ghostState.scaredTimer for ghostState in cur_ghost_states]

    cur_capsule = currentstate.getCapsules()

    #take the new score and the current so we can calculate the dif
    #newScore = successorstate.getScore()
    cur_score = scoreEvaluationFunction(currentstate)
    score = cur_score

    #find the closest ghost
    closer_ghost = min([manhattanDistance(cur_pos, ghost.getPosition()) for ghost in cur_ghost_states])

    #find closest food
    if cur_food:
        closeFoodDist = min([manhattanDistance(cur_pos, food) for food in cur_food])
    else:
        closeFoodDist = 0

    #see if we can eat ghosts
    smallScareTime = min(cur_scared_times)
    if smallScareTime != 0:
        closer_ghost = 0
        
    #the grater the food dist the smaller the score
    ret_num = (10/(closeFoodDist+1)) #plus one so its not divided with zero
    #the further the ghost is the greater the score
    ret_num = ret_num +(closer_ghost/10)
    ret_num = ret_num + score
    return ret_num


    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
