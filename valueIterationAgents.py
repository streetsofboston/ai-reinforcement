# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

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

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allStates = mdp.getStates()
        
        for iter in range(0, iterations+1):
            valuesIter = self.calcValue(allStates, iter)
            self.values = valuesIter.copy()

    def calcValue(self, allStates, iteration):
        mdp = self.mdp;
        
        valuesK = util.Counter()
        
        for state in allStates:
            if iteration == 0:
                valuesK[state] = 0
            else:
                qvalues = [self.getQValue(state, action) for action in mdp.getPossibleActions(state)]
                valuesK[state] = max(qvalues) if len(qvalues) > 0 else self.values[state]
        
        return valuesK
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        # For value iteration, this will return the V(k) when calculating V(k+1).
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        gamma = self.discount
        successors = mdp.getTransitionStatesAndProbs(state, action)
        
        return sum([successor[1] * (mdp.getReward(state, action, successor[0]) + gamma * self.getValue(successor[0])) for successor in successors])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        
        qvalues = [(action, self.getQValue(state, action)) for action in actions]
        maxQValue = max(qvalues, key=lambda item: item[1])
        
        return maxQValue[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        # For value iteration, this will return the Q(k) when calculating Q(k+1).
        return self.computeQValueFromValues(state, action)
    
       
