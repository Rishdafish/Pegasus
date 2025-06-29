import random
from .prompts import prompts
from .worker import agent


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
        for agentidx in range(self.StartingAgents):
            agentRole = TotalRoles[agentidx]
            random_idx = random.randint(0, len(Tasks) - 1)
            agentTask = Tasks[random_idx]
            currAgent = agent(agentRole, agentTask, isFirstGen=True)
            currAgent.evaluate()
            self.currentAgents.append(currAgent)
    
    
    def mergeAgents(self):
        pass

    def runNextGen(self):
        if self.generation <= self.maxGeneration:
            for agent in self.currentAgents:
                self.nextGeneration = agent.createChildren()


    def run(self):
        if self.generation == 0:
            self.FirstGenerationStart()
        

        # Evaluate agents and prepare for next generation
        self.evaluateAgents()
        self.generation += 1
    def scoreAgent(self):
        pass 


'''
Sampling Parameters (LLM Call):
- temperature = 1.3
- top_p = 0.95
- presence_penalty = 0.7
- repeat_penalty = 1.1

'''
