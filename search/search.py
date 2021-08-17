# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Queue
from util import PriorityQueue
from util import Stack

debug = True

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    #This should be the starting state of the problem I believe
    startState = problem.getStartState()

    

    #print("Start:", startState)
    #print("Is the start a goal?", problem.isGoalState(startState))
    #print("Start's successors:", problem.getSuccessors(startState))

    "*** YOUR CODE HERE ***"
    
    #This was the original algorithm. It finds the path but doesn't record it.
    #I altered it to make it so I could record it
    """
    #This is the stack that holds all the states
    
    #It has to be a stack as it is depth first search,
    #so the last node opened is the last explored
    stateStack = Stack()
    stateStack.push(startState)
    
    #We make an empty set for visited so we can fast check if visited
    #possesses the current item
    visited = {}
    
    #Holds the currently followed path
    #We make it a stack as it needs to be able to remove the
    #most recent node visited if it's not along the path to the goal
    currentPath = []

    #This holds the currentNode being evaluated
    #It's not really a node but the state, however node is easier to understand
    currentNode = stateStack.pop()
    
    #This is the while loop for the dfs that allows us to access all
    #nodes until we reach the goal state
    while problem.isGoalState(currentNode) == False:

        #If the current node has not been visited, operate on it
        if currentNode not in visited:
            
            #Get all the children
            children = problem.getSuccessors(currentNode)

            #iterate over all children and handle them
            for child in children:
                
                #This is what they called it in searchAgent.py, so that's what I'm gonna call it
                nextState, action, cost = child
                

                # If the child's state has not been visited, visit it
                if nextState not in visited:
                    
                    #Add the action to the current path

                    #Add the nextState to the state stack
                

            #Mark the currentNode as visited and then set the new current node
            visited.add(currentNode)

            currentPath, currentNode = stateStack.pop()

            

    #This converts the currentPath Stack into an array to return
    returner = []
    while currentPath.isEmpty() == False:
        returner.append(currentPath.pop())

    #The return statement
    return returner
    """
    #I'm gonna hold each state in the visited stack but I will record
    #the path to the location and the cost of said path to the array
    #So each item will be (state, pathToState, costArrayForEachDirection)
    pathHolder = []
    cost = 0
    

    #Holds all the nodes that have been visited
    visited = []

    #This holds the states, path's to the state, and the cost's to the states that have been found
    nodeStack = Stack()
    
    #Add the first item to the stack
    nodeStack.push( (startState, pathHolder, cost) )    

    #Holds the temps that get the Nodes of the state
    while nodeStack.isEmpty() == False:
        #Get the next node in the state stack
        currentState, currentPath, currentCost = nodeStack.pop()
        
        #Check to see if the current state has been visited before
        #if has not been visited, handle it
        #else ignore it
        if currentState not in visited:
            #Add it to the visited node set
            visited.append(currentState)

            #If the currentNode's state is the goal state, return the path to the current node
            if problem.isGoalState(currentState):
                return currentPath

            #Add all of it's children with their path's and their costArrays
            #to the state stack
            for child in problem.getSuccessors(currentState):
                
                # Need to


                #Get all the values seperated
                childState, childDirection, costToChild = child
                
                #Add the new child with it's direction appended to the array and the cost added
                #Creates the new sub items of the nodes
                childPath = currentPath + [childDirection]
                childCost = currentCost + costToChild

                nodeStack.push( ( childState , childPath, childCost) )

    #If it gets here, that means the goalState is not accessable from the currentState and you fucked up somehow
    if debug == True:
        print(visited)
    # So return an empty path
    #return []

    #DEBUG ONLY
    if debug == True:
        print(visited)
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # I just copied the depth first search and replaced the stack with a queue
    

    #Forgot this line, lol
    startState = problem.getStartState()

     #I'm gonna hold each state in the visited stack but I will record
    #the path to the location and the cost of said path to the array
    #So each item will be (state, pathToState, costArrayForEachDirection)
    pathHolder = []
    cost = 0


    #Holds all the nodes that have been visited
    visited = []

    #This holds the states, path's to the state, and the cost's to the states that have been found
    nodeQueue = Queue()

    #Add the first item to the stack
    nodeQueue.push( (startState, pathHolder, cost) )


    #Holds the temps that get the Nodes of the state
    while nodeQueue.isEmpty() == False:
        #Get the next node in the state stack
        currentState, currentPath, currentCost = nodeQueue.pop()

        #Check to see if the current state has been visited before
        #if has not been visited, handle it
        #else ignore it
        if currentState not in visited:
            #Add it to the visited node set
            visited.append(currentState)

            #If the currentNode's state is the goal state, return the path to the current node
            if problem.isGoalState(currentState):
                return currentPath

            #Add all of it's children with their path's and their costArrays
            #to the state stack
            for child in problem.getSuccessors(currentState):
                # DEBUGGER ONLY
                #print("\nThe child looks like:\n")
                #print(child)
                #print("\n")
                
                
                #Get all the values seperated
                childState, childDirection, costToChild = child

                #Add the new child with it's direction appended to the array and the cost added
                #Creates the new sub items of the nodes
                childPath = currentPath + [childDirection]
                childCost = currentCost + costToChild

                nodeQueue.push( ( childState , childPath, childCost) )

    #If it gets here, that means the goalState is not accessable from the currentState and you fucked up somehow
    # So return an empty path
    return []

    
    
    
    
    
    
    
    
    
    
    
    
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    
    #Gets the initial stateState of the problem
    #Problem will never be handled from here on out
    startState = problem.getStartState()
    
    #Need to hold the paths, the visited nodes, etc
    pathHolder = []
    visited = []
    cost = 0

    #Sets up the priority queue
    pq = PriorityQueue()
    
    #Sets up the first item in the priority queue
    currentNode = (startState, pathHolder, cost)
    #currentNode = startState
    #print(currentNode)

    #Get the first state and give it cost 0
    pq.push(currentNode, cost)
    #pq.push( (startState, pathHolder, cost), cost)

    #Handle every item in the priority queue
    while pq.isEmpty() == False:
        
        #Get the first item in the queue
        currentNode = pq.pop()
        
        #print(currentNode)

        currentState, currentPath, currentCost = currentNode
    
        #If the goalstate has been reached, return the current state the current path
        if problem.isGoalState(currentState):
            return currentPath

        #Else, handle the pathing of the current node if it has not been visited
        if currentState not in visited:
            
            #First add the current node to the visited stack
            visited.append(currentState)

            #Add all of it's children with their path's and their costArrays
            #to the state stack
            for child in problem.getSuccessors(currentState):
                
                #Get all the sub values of the child array
                childState, childDirection, costToChild = child
                
                #Make the new metadata
                childPath = currentPath + [childDirection]
                childCost = currentCost + costToChild
                
                node = (childState, childPath, childCost)

                #Add it to the priority queue
                pq.push(node, childCost)
        
    #If it gets here, that means the goalState is not accessable from the currentState and you fucked up somehow
    # So return an empty path
    return []

    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #Gets the initial stateState of the problem
    #Problem will never be handled from here on out
    startState = problem.getStartState()

    #Need to hold the paths, the visited nodes, etc
    pathHolder = []
    visited = []
    cost = 0

    #Sets up the priority queue
    pq = PriorityQueue()

    #Sets up the first item in the priority queue
    currentNode = (startState, pathHolder, cost)
    #currentNode = startState
    #print(currentNode)

    #Get the first state and give it cost 0
    pq.push(currentNode, cost)
    #pq.push( (startState, pathHolder, cost), cost)

    #Handle every item in the priority queue
    while pq.isEmpty() == False:
        
        #Get the first item in the queue
        currentNode = pq.pop()
        #print(currentNode)
        currentState, currentPath, currentCost = currentNode
            
        #Handle it if it's the goal
        if problem.isGoalState(currentState):
            return currentPath

        #Else, handle the pathing of the current node if it has not been visited
        if currentState not in visited:

            #First add the current node to the visited stack
            visited.append(currentState)

            #Add all of it's children with their path's and their costArrays
            #to the state stack
            for child in problem.getSuccessors(currentState):

                #Get all the sub values of the child array
                childState, childDirection, costToChild = child

                #Make the new metadata
                childPath = currentPath + [childDirection]
                childCost = currentCost + costToChild

                node = (childState, childPath, childCost)

                #Add it to the priority queue
                pq.push(node, childCost + heuristic(childState,problem))
 
    #If it gets here, that means the goalState is not accessable from the currentState and you fucked up somehow
    # So return an empty path
    return []

    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
