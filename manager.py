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
                 nextGeneration = [],
                 childrenPerGeneration = 2):
        

        self.currentAgents = currentAgents        
        self.maxGeneration = maxGenerations
        self.generation = generation
        self.MaxAgents = MaxAgents 
        self.task = task
        self.nextGeneration = nextGeneration
        self.best_agents = best_agents
        self.StartingAgents = StartingAgents
        self.childrenPerGeneration = childrenPerGeneration

    def FirstGenerationStart(self):
        self.generation = 1
        TotalRoles = prompts.themes
        Tasks = prompts.tasks
        temp = []
        for agentidx in range(self.StartingAgents):
            agentRole = TotalRoles[agentidx]
            random_idx = random.randint(0, len(Tasks) - 1)
            agentTask = Tasks[random_idx]
            currAgent = agent(agentRole, agentTask, isFirstGen=True)
            currAgent.evaluate()
            self.scoreAgent(currAgent)
            temp.append(currAgent)
        
    
    def mergeAgents(self):
        pass

    def runNextGen(self):
        if self.generation <= self.maxGeneration:
            newAgents = []
            for parent in self.currentAgents:
                child1, child2 = agent.createChildren(parent, self.childrenPerGeneration)
                self.scoreAgent(child1)
                self.scoreAgent(child2)
                newAgents.append(child1, child2)
            self.generation += 1
            self.currentAgents = newAgents
            if len(self.currentAgents) > self.MaxAgents:
                self.cutDown()
            else: 
                self.runNextGen()

    def cutDown(self):
        pass

    def run(self):
        if self.generation == 0:
            self.FirstGenerationStart()
        

        # Evaluate agents and prepare for next generation
        self.evaluateAgents()
        self.generation += 1
    def scoreAgent(self, agent: Agent() ):
        pass 


'''
Sampling Parameters (LLM Call):
- temperature = 1.3
- top_p = 0.95
- presence_penalty = 0.7
- repeat_penalty = 1.1

'''
