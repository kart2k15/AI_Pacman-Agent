# pacmanAgents.py
# ---------------
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
#BFSAgent with list modified to work as FIFO Queue
#A*_Agent with in_house priority queue

from pacman import Directions
from game import Agent
from heuristics import *
import random

import sys




class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        # evaluate the successor states using scoreEvaluation heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        # get best choice
        bestScore = min(scored)[0]
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        # return random action from the list of the best actions
        return random.choice(bestActions)

class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write BFS Algorithm instead of returning Directions.STOP
        current_node=state

        frontier = []
        total_cost_source=0+admissibleHeuristic(state)

        frontier.append([current_node, "no-action-root", 0, total_cost_source,[""]])

        explored = []
        z=0
        d=-1
        while len(frontier)>0:
            node_tup=frontier.pop(0)
            explored.append(node_tup)
            for action in node_tup[0].getLegalPacmanActions():
                child=node_tup[0].generatePacmanSuccessor(action)
                if(child==None):
                    min_node = ("", "", "", "")

                    total_min_cost = sys.maxint

                    for node_tuple in frontier:
                        if (total_min_cost > node_tuple[3]):
                            min_node = node_tuple
                    return min_node[4][1]



                else:
                    action_that_lead_to_child=action

                    action_l=node_tup[4] + [action_that_lead_to_child]

                    cost_from_source=node_tup[2]+1
                    total_cost_child=cost_from_source +admissibleHeuristic(child)



                    if ((child not in frontier) or (child not in explored)):
                        if child.isWin():
                            return action_l[1]


                frontier.append([child, action_that_lead_to_child, cost_from_source, total_cost_child,action_l])







class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        current_node = state

        frontier = []
        total_cost_source = 0 + admissibleHeuristic(state)

        frontier.append([current_node, "no-action-root", 0, total_cost_source, [""]])

        explored = []
        z = 0
        d = -1
        while len(frontier) > 0:
            node_tup = frontier.pop()
            explored.append(node_tup)
            for action in node_tup[0].getLegalPacmanActions():
                child = node_tup[0].generatePacmanSuccessor(action)
                if (child == None):
                    min_node = ("", "", "", "")

                    total_min_cost = sys.maxint

                    for node_tuple in frontier:
                        if (total_min_cost > node_tuple[3]):
                            min_node = node_tuple
                    return min_node[4][1]




                else:
                    action_that_lead_to_child = action

                    action_l = node_tup[4] + [action_that_lead_to_child]

                    cost_from_source = node_tup[2] + 1
                    total_cost_child = cost_from_source + admissibleHeuristic(child)

                    if ((child not in frontier) or (child not in explored)):
                        if child.isWin():
                            return action_l[1]

                frontier.append([child, action_that_lead_to_child, cost_from_source, total_cost_child, action_l])


class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame

    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        source=state
        source_depth_cost=0
        source_heuristic_cost=admissibleHeuristic(state)
        action_list=[""]
        state_PQ=[[source,source_depth_cost,source_heuristic_cost,action_list]]
        explored=[]

        def pq_pop_min_cost():
            min_node = None
            min_path_cost = sys.maxint
            min_heuristic_cost = sys.maxint
            min_node_index = -1
            min_total_cost = sys.maxint
            for x, y in enumerate(state_PQ):
                sum = y[1] + y[2]
                if (sum <= min_total_cost):
                    min_total_cost = sum
                    min_node = y[0]
                    min_node_index = x
                    min_path_cost = y[1]
                    min_heuristic_cost = y[2]





            return [state_PQ[min_node_index], min_node_index]

        while len(state_PQ)>0:
            #the minimum node_list in node_tup
            node_tup=pq_pop_min_cost()[0]
            #"actually" deleting that node from the state_PQ (our priority queue)
            if(len(state_PQ)>0):
                state_PQ.pop(node_tup[1])
            if(node_tup[0].isWin()):
                return node_tup[3][0]
            explored.append(node_tup[0])
            for action in node_tup[0].getLegalPacmanActions():
                child=node_tup[0].generatePacmanSuccessor(action)
                if(child==None):
                    min_node_tup=pq_pop_min_cost()[0]
                    return min_node_tup[3][1]
                else:
                    action_child=action
                    child_action_l=node_tup[3]+[action_child]
                    child_path_cost=node_tup[1]+1
                    child_heuristic_cost=admissibleHeuristic(child)
                    child_total_cost=child_path_cost+child_heuristic_cost
                    if (child not in explored) or([child,child_path_cost,child_heuristic_cost,child_action_l] in state_PQ):
                        state_PQ.append([child,child_path_cost,child_heuristic_cost,child_action_l])
                    elif ([child,child_path_cost,child_heuristic_cost,child_action_l] in state_PQ):
                        for x, node_l in enumerate(state_PQ):
                            if(node_l[0]==child) and ((node_l[1]+node_l[2])>child_total_cost):
                                state_PQ[x]=[child,child_path_cost,child_heuristic_cost,child_action_l]



