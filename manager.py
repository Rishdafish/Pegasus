import ast 
import string
import secrets
import random
from .prompts import prompts
from .worker import agent
import os
from openai import OpenAI


class Manager: 
    def __init__(self, task, 
                 MaxAgents, 
                 StartingAgents = 200, 
                 generation = 0, 
                 maxGenerations = 15, 
                 currentAgents = [],
                 best_agents = [], 
                 mixedGeneration = [],
                 childrenPerGeneration = 2):
        

        self.currentAgents = currentAgents        
        self.maxGeneration = maxGenerations
        self.generation = generation
        self.MaxAgents = MaxAgents 
        self.task = task
        self.mixedGeneration = mixedGeneration
        self.best_agents = best_agents
        self.StartingAgents = StartingAgents
        self.childrenPerGeneration = childrenPerGeneration
          
    def generate_unique_id(self, length=6):
        alphabet = string.ascii_uppercase + string.digits
        while True:
            new_id = ''.join(secrets.choice(alphabet) for _ in range(length))
            for agent in self.currentAgents:
                if agent.getId() == new_id:
                    self.generate_unique_id()
            return new_id 
                
    def FirstGenerationStart(self):
        self.generation = 1
        TotalRoles = prompts.themes
        Tasks = prompts.tasks
        temp = []
        for agentidx in range(self.StartingAgents):
            agentRole = TotalRoles[agentidx]
            random_idx = random.randint(0, len(Tasks) - 1)
            agentTask = Tasks[random_idx]
            identification = self.generate_unique_id()
            currAgent = agent(agentRole, agentTask, isFirstGen=True, id = identification)
            self.generateCode(currAgent)
            self.scoreAgent(currAgent)
            temp.append(currAgent)
        self.currentAgents = temp
        
    def visualize(self): 
        pass 


    def merge_agents(self):
            """
            1) Generate 25 new agents by merging pairs (roulette‐wheel) from self.best_agents via the LLM.
            2) Remove the bottom 25 of self.currentAgents (assumes list already sorted by score desc).
            3) Extend self.currentAgents with the 25 new agents and store them in self.mixedGeneration.
            """
            OFFSPRING_COUNT = 25
            children = []

            # helper: fitness‐proportionate selection
            def pick_two_by_fitness(agents):
                total = sum(a.getScore() for a in agents)
                if total == 0:
                    return random.sample(agents, 2)
                def pick_one():
                    r = random.uniform(0, total)
                    cum = 0.0
                    for ag in agents:
                        cum += ag.getScore()
                        if cum >= r:
                            return ag
                    return agents[-1]
                return pick_one(), pick_one()

            # 1) generate offspring
            for _ in range(OFFSPRING_COUNT):
                pa, pb = pick_two_by_fitness(self.best_agents)
                role, merged_code = ast.literal_eval(self.llmCall(isMerged = True)) #returns a list the merged code role 
                Tasks = prompts.tasks
                random_idx = random.randint(0, len(Tasks) - 1)
                agentTask = Tasks[random_idx]
                child = agent(code = merged_code, id = self.generate_unique_id(), parents=[pa.getId(), pb.getId()],task=agentTask, role=role)
                children.append(child)
            survivors = self.currentAgents[: self.MaxAgents - OFFSPRING_COUNT]

            # 3) form new generation
            self.mixedGeneration = survivors + children
            # update currentAgents for next round
            self.currentAgents  = self.mixedGeneration
            # clear best_agents so it can be recomputed after evaluation
            self.best_agents = []

    def runNextGen(self):
        if self.generation <= self.maxGeneration:
            newAgents = []
            for parent in self.currentAgents:
                child1, child2 = agent.createChildren(parent, self.childrenPerGeneration)
                self.generateCode(child1)
                self.generateCode(child2)
                self.scoreAgent(child1)
                self.scoreAgent(child2)
                newAgents.append(child1, child2)
            self.generation += 1
            self.currentAgents = newAgents
            #  #Getting the first 50, which are the best according to their score
            if self.generation > 3 and len(self.currentAgents * 2 ) >= self.MaxAgents:
                self.currentAgents.sort(key=agent.getScore(), reverse=True)
                self.best_agents = self.currentAgents[:50]
                self.mergeAgents()
            if len(self.currentAgents) > self.MaxAgents:
                self.trim()
                self.runNextGen()
            else: 
                self.runNextGen()
        else:
            print("Max generations reached. Stopping evolution.")
            self.visualize()
            return

    def trim(self):
            """
            Sorts agents descending by score, then keeps only the top self.maxAgents.
            """
            total = len(self.currentAgents)
            # If we have more than allowed, trim the worst
            if total > self.MaxAgents:
                # Sort best→worst by score
                self.currentAgents.sort(key=lambda agent: agent.score, reverse=True)
                # Keep only the top maxAgents
                self.currentAgents = self.currentAgents[: self.MaxAgents]
                print(f"Trimmed population from {total} → {len(self.currentAgents)} agents.")
            else:
                print(f"No trim needed ({total} ≤ {self.MaxAgents}).")


    def run(self):
        if self.generation == 0:
            self.FirstGenerationStart()
        

        # Evaluate agents and prepare for next generation
        self.evaluateAgents()
        self.generation += 1

    def scoreAgent(self, Agent: agent):
        pass 

    def generateCode(self, Agent: agent):
        pass

    def llmCall(self, isMerged = False : bool): 
    pass
    


'''
Sampling Parameters (LLM Call):
- temperature = 1.3
- top_p = 0.95
- presence_penalty = 0.7
- repeat_penalty = 1.1

'''
