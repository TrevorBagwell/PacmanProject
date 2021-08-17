# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.values = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        #Holds the maximum value
        maxval = 0.0
        
        # If there exists a legal action, find its value
        if len(self.getLegalActions(state)) != 0:
            # Reupdate the maxval so that it's the absolute minimum possible float value
            maxval = -float('inf')

            # Iterate through the values and take the larger one
            for action in self.getLegalActions(state):
                # Takes the larger of the 2 values
                maxval = max(maxval,self.getQValue(state,action))
        
        # Returns the maximum value
        return maxval
        
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # This holds the current value
        value = self.getValue(state)

        # FInd the corresponding action to the current value
        for action in self.getLegalActions(state):
            # If the value is equal to the qValue, its the action
            if value == self.getQValue(state,action):
                return action

        return None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # Handle a coin flip with weight epsilon
        if util.flipCoin(self.epsilon):
        
            # Good tip ngl
            action = random.choice(legalActions)
        
        # Else get the proper action
        else:
        
            # Get the policy for the state
            action = self.getPolicy(state)

        # Return the action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        
        # Get the alpha value
        alpha = self.alpha
        # Get the gamma value
        gamma = self.discount
        # Get the v' value
        value = self.getValue(nextState)
        # Get the current qValue
        qValue = self.getQValue(state,action)
        
        # Held for debug purposes
        temp = (1 - alpha)*qValue + alpha * ( reward + gamma * value ) 
        
        # Update the value
        self.values[(state,action)] = temp
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        # This holds the features from the feature extractor that we were told about in the Q10 explanation
        feats = self.featExtractor.getFeatures(state,action)

        # Holds the current running q value
        qValue = 0.0

        # Sum to get the qValue
        # So turns out we want to iterate over the keys of feats and not all the weights
        # And I'm just an idiot and it took me a super long time to literally reread the question
        # where it even notes this as a necessity for solution
        # I'm max dumbo
        for key in feats.keys():
            # Add each feature*weight
            qValue += feats[key]*self.weights[key]

        return qValue
        


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        
        # This holds the features from the feature extractor that we were told about in the Q10 explanation
        feats = self.featExtractor.getFeatures(state,action)
        # Holds the alpha of self.alpha
        alpha = self.alpha
        # Holds the discount as gamma to fit with the equation
        gamma = self.discount
        # This is the difference we were told about to use instead
        difference = (reward + gamma * self.getValue(nextState)) - self.getQValue(state,action)

        
        # Update every single weight by adding alpha * difference * feature[i](state,action)
        for key in self.weights.keys():
            # Just do what the question instructs
            self.weights[key] = self.weights[key] +  alpha * difference * feats[key]
        


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #TODO:
            pass
