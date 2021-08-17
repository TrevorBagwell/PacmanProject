# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        #DEBUG PURPOSES ONLY
        #self.debug = True
        if True:
            print("\n\n\n\n\n")



    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # We need the set of all state, actions, the state transiton function
        # The reward function and a threshold
        # > The states are stored in self.mdp
        # > The reward function can be handled from self.mdp
        # > The actions can be derived from each state using self.mdp
        # > The state transition function can be derived from self.mdp
        # > The threshold is self.iterations
        #TODO:
        # This holds all the states in the passed in mdp
        states = self.mdp.getStates()
        
        # K hold's the current iteration count and is limited by the total iterations we want
        # to run
        # This line runs value iteration the number of times indicated by the initialization
        for k in range(self.iterations):
            # We need a holder for the K+1 items so it doesn't think we want to calculate an action from the current values list as we could update a value before we use the intended value
            V_k = util.Counter()
            # Iterate over all the states and earn urself a free scone
            # Jk, we need to handle each state so we just iterate it
            for state in states:
                # Set all terminal states to a value of 0
                if self.mdp.isTerminal(state):
                    V_k[state] = 0
                    continue

                # Handle the instance in which it's not terminal
                # We need to find all the maximum of all the q values
                # This value holds the maximum value
                # Initialized to -(2^31-1) as thats the max unsigned int on a 
                # 32 bit system
                qvalues = []
            
                # Iterate over all the actions and calculate their q value
                for action in self.mdp.getPossibleActions(state):
                    # This holds the q value for the tiems
                    qvalues.append(self.computeQValueFromValues(state,action))
                
                # Add the value to the values dictionary
                if not qvalues:
                    continue
                else:
                    V_k[state] = max(qvalues)
                
                if False and k > self.iterations-2:
                    print("finds max value ")
                    if not qvalues:
                        print("none")
                    else:
                        print(max(qvalues))
                    print(" for ")
                    print(state)
                    print(" which says go: ")
                    print(self.computeActionFromValues(state))
                    print("\n")
            # We set the value of values to the value of V_k
            self.values = V_k
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #TODO:
        # This holds the q value for the tiems
        qValue = 0

        # Add all the values of the transitions to the qvalue
        for transition in self.mdp.getTransitionStatesAndProbs(state,action):
            # This holds the transition prob value
            # This is what mdp.py says is the probability
            p = transition[1]
            # This holds the reward value for our item
            # This is calculated using the mdp's function passed in with the args
            r = self.mdp.getReward(state,action,transition[0])
            #This holds the discount
            gamma = self.discount
            # This is the previous iterations value
            v = self.values[transition[0]]

            qValue += p*(r+gamma*v)
            


            if False:
                c = ","
                output = str("p,r,gamma,v, qValue: ") +str(p)+c+str(r)+c+str(gamma)+c+str(v)+c+str(qValue)
                print(output)

        return qValue

        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #TODO:
        # Iterate over all the actions and find the action with the highest qvalue
        # for the state
        
        desiredAction = None

        # If the state is terminal, there exists no action possible    
        if self.mdp.isTerminal(state):
            return desiredAction
        
        # Iteration loop
        for action in self.mdp.getPossibleActions(state):
            # Get the qvalue for the action
            qValue = self.computeQValueFromValues(state,action)
            
            #DEBUG ONLY
            if False:
                output = "The action "+str(action)+" has q value "
                output += str(qValue) + " while the state has value "
                output += str(self.values[state])

            # Set the desired action and break if we reach the desired action
            if self.values[state] <= qValue:
                desiredAction = action

        return desiredAction
        
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #TODO:



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #TODO:
