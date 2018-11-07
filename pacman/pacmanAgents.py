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
#BFSAgent with list modified to work as Queue
#DFSAgent with list modified to work as Stack
#AStarAgent(A*)-- used a 2D array & implemented Priority Queue to maintain total minimum cost through list-comprehensions

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

class RandomSequenceAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        self.actionList = [];
        for i in range(0,10):
            self.actionList.append(Directions.STOP);
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman

        possible = state.getAllPossibleActions();
        print len(possible)
        for i in range(0,len(self.actionList)):
            self.actionList[i] = possible[random.randint(0,len(possible)-1)];
        tempState = state;

        for i in range(0,len(self.actionList)):
            if tempState.isWin() + tempState.isLose() == 0:
                tempState = tempState.generatePacmanSuccessor(self.actionList[i]);
            else:
                break;
        # returns random action from all the valid actions
        return self.actionList[0];

class HillClimberAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        possible = state.getAllPossibleActions()

        #function below scores/ranks action sequence
        def evaluate_seq(action_seq):
            last=-1
            #last is technically the last state, after executing action seq
            #abused python's dynamic environment
            next=-1
            for action in action_seq:
                next=state.generatePacmanSuccessor(action)

                if(next==None):
                    return -4
                elif next.isWin() + next.isLose() == 0:
                    break;
            last=next
            evaluation=gameEvaluation(state,last)
            return evaluation
        #on a side note---no need to check for terminal state as the probability of reaching it is
        #negligible, further even if one does reach such a state---it doesn't really matter
        #because no matter what kind of state we reach [while executing an action sequence]
        #be it terminal/non-terminal-----
        # the getAction method for RandomSequenceAgent,GeneticAgent, & HillClimberAgent always returns
        #the first action of the best action sequence found till now. And obviously we check for the
        # limiting condition of generating successor nodes

        #function below changes actions in action_seq with 50% probability
        def rand_action_seq(action_seq):
            for i,action in enumerate(action_seq):
                x=random.randint(0,1)
                if x==1:
                    action_seq[i]=possible[random.randint(0,len(possible)-1)]
            return action_seq



        first_action_seq=[]
        for i in range(0,5):
            first_action_seq.append(Directions.STOP)
        for i in range(0,5):
            first_action_seq[i]=possible[random.randint(0,len(possible)-1)]
        first_evaluation=evaluate_seq(first_action_seq)
        current_best_action_seq=first_action_seq
        current_best_evaluation=first_evaluation
        best=[current_best_action_seq,current_best_evaluation]

        for i in range(0,3000):
            next_seq=rand_action_seq(current_best_action_seq)
            next_eval=evaluate_seq(next_seq)
            if(next_eval==-4):
                return current_best_action_seq[0]
            elif(next_eval>current_best_evaluation):
                current_best_action_seq=next_seq
                current_best_evaluation=next_eval
                best=[current_best_action_seq, current_best_evaluation]




        # TODO: write Hill Climber Algorithm instead of returning Directions.STOP



class GeneticAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        possible=state.getAllPossibleActions()


        global best_evaluation_till_now
        best_evaluation_till_now=-sys.maxint

        best_action_seq_till_now = []
        for i in range(0,5):
            best_action_seq_till_now.append(Directions.STOP)
        # TODO: write Genetic Algorithm instead of returning Directions.STOP

        def evaluate_seq (action_seq):
            last=-1
            # last is technically the last state, after executing action seq
            # abused python's dynamic environment
            next=-1

            for action in action_seq:
                next=state.generatePacmanSuccessor(action)
                if(next==None):
                    return [-4,-4]
                elif next.isWin() + next.isLose() == 0:
                    break;
            last=next

            # on a side note---no need to check for terminal state as the probability of reaching it is
            # negligible, further even if one does reach such a state---it doesn't really matter
            # because no matter what kind of state we reach [while executing an action sequence]
            # be it terminal/non-terminal-----
            # the getAction method for RandomSequenceAgent,GeneticAgent, & HillClimberAgent always returns
            # the first action of the best action sequence found till now. And obviously we check for the
            # limiting condition of generating successor nodes


            # since we return the 1st action of best action sequence among all generations. we keep track of the best_action_sequence_here
            evaluation=gameEvaluation(state,last)
            global best_evaluation_till_now
            if(evaluation>best_evaluation_till_now):
                best_evaluation_till_now=evaluation
                for i in range(len(action_seq)):
                    best_action_seq_till_now[i]=action_seq[i]

            return [evaluation, action_seq]


        def get_two_offsprings_from_current_generation(generation):
            l = []  # l stores action_seq and it's evaluations
            for i in range(0, 8):
                eval_list = evaluate_seq(generation[i])
                if(eval_list==[-4,-4]):
                    return [-10,-10]
                l.append(eval_list)



                for i in range(0, len(l) - 1):
                    for j in range(0, len(l) - i - 1):
                        if (l[j][0] > l[j + 1][0]):
                            temp = l[j]
                            l[j] = l[j + 1]
                            l[j + 1] = temp
            #now we have sorted l according to evaluations of each seq
            #now we assign ranks
            rank = 1
            seq_rank = []  # this seq contains action_seq and rank together
            for i in range(0, len(l)):
                seq_rank.append([rank, l[i][1]])
                rank = rank + 1

            #now below is the part the code that makes sure fitter chromosomes have a higher probability
            #of getting selected nonetheless there is significant of chance of a lesser fitting node to be selected
            #& selects two chromsomes
            arr = [8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1]

            '''How? Well, arr contains 8 eights, 7 sevens and 6 sixes and so forth. And x is random a integer generated b/w 0<=x<=35; x corresponds
            to an index in arr. Now the rank of the chromosome selected is arr[x].
            
            Proof of Correctness---
            Without loss of any generality,
            8 clearly has a higher probability of being selected, since it has the highest frequency/mode. Probability of chromosomes (chm) 
            *P(8th_rank_chm)=8/36, P(7th_rank_chm)=7/36 & so forth. Thus for any arbitary rank r of a given chromosome in a given population/generation, 
            where r is a positive integer. 
            
            P(rth_Rank_Chromosome)= r/36 
            
            Further we make sure
            no two chromosomes have the same rank. For the 2nd rank, we keep generating another chromosome while making sure it isn't equal to the previous
            chromosome (this step ensures uniqueness of ranks/chromosomes [i.e. the two selected ranks/chm are distinct], thus each r is distinct, 
            hence a set of ranks NOT a multiset of ranks. 
            
            '''


            x = random.randint(0, 35)
            chromosome1_rank = arr[x]
            chromosome1 = [Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP]
            for i in range(0, len(seq_rank) - 1):
                if seq_rank[i][0] == chromosome1_rank:

                    for x in range(0,len(seq_rank[i][1])):
                        chromosome1[x]=seq_rank[i][1][x]

            chromosome2_rank = -1
            chromosome2 = [Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP]
            while True:
                x1 = random.randint(0, 35)
                #to check if we don't get the same parent again
                if (arr[x1] != arr[x]):
                    chromosome2_rank = arr[x1]
                    break

            for i in range(0, len(seq_rank) - 1):
                if seq_rank[i][0] == chromosome2_rank:

                    for x in range(0,len(seq_rank[i][1])):
                        chromosome2[x]=seq_rank[i][1][x]


            #now create two offsprings
            # we have to start the cross-over process
            z = random.randint(1, 10)
            offspring1_seq = [Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP, Directions.STOP]

            offspring2_seq = [Directions.STOP,Directions.STOP,Directions.STOP,Directions.STOP, Directions.STOP]
            if (z <= 7):
                # implies we do cross-over for <=70 percent

                for i in range(0, len(chromosome1) - 1):
                    a = random.randint(0, 1)
                    # if a is 0 we get gene from chromosome1 if it's 1 we get gene from chromosome2 for offspring1
                    # if a is 0 we get gene from chromosome2 if it's 1 we get gene from chromosome1 for offspring2
                    if (a == 0):
                        offspring1_seq[i]=chromosome1[i]
                        offspring2_seq[i]=chromosome2[i]
                    if (a == 1):
                        offspring1_seq[i]=chromosome2[i]
                        offspring2_seq[i]=chromosome1[i]
            elif z > 7:
                offspring1_seq = chromosome1
                offspring2_seq = chromosome2
                # that is if test>70% we just include parents in new generation

            return [offspring1_seq,offspring2_seq]


        first_generation=[]
        action_seq=[]



        #Now actual initialization of first generation
        for i in range(0,8):
            chromosome = []
            #initializing each chromosome present in 1st generation i.e. seq
            for i1 in range(0,5):
                randomNumberGenerated = random.randint(0,len(possible)-1)
                chromosome.append(possible[randomNumberGenerated])
            first_generation.append(chromosome)

        #Finally we have the first generation FULLY initialized

        current_generation=first_generation
        #we keep manipulating current generation only, that is we keep track of one generation/population only instead of all of them
        for x in range(0,3000):
            #now we start the genetic algo
            offspring_generation=[] #that is the new generation
            for i in range(0,4):
                offspring_list=get_two_offsprings_from_current_generation(current_generation)
                if offspring_list==[-10,-10]:
                    return best_action_seq_till_now[0]
                #add offsprings to the offspring generation thus generating a new population
                offspring_generation.append(offspring_list[0])
                offspring_generation.append(offspring_list[1])
            current_generation=offspring_generation

            #now we have the new population stored in current_generation
            #need to perform mutations
            for y,z in enumerate(current_generation):
                f=random.random()
                if f<=0.1:
                    #implies we mutate

                    chromosome_to_mutate= current_generation[y]
                    g=random.randint(0,len(chromosome_to_mutate)-1)
                    #g is index of gene to be mutated
                    chromosome_to_mutate[g]=chromosome_to_mutate[random.randint(0,len(chromosome_to_mutate)-1)]
                    mutated_chromosome=chromosome_to_mutate
                    #now chromosome has been mutated
                    #now we replace the chromosome present in the population, with a mutated version of itself
                    current_generation[y]=mutated_chromosome


class MCTSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write MCTS Algorithm instead of returning Directions.STOP
        return Directions.STOP




