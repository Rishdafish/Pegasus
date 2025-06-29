from .manager import Manager 
from .prompts import prompts 
class agent(): 
    def __init__(self, role, task, isFirstGen = False): 
        super().__init__()
        self.role = role
        self.task = task
        self.evaluation = None
        self.parents = []
        self.isFirstGen = isFirstGen


    def setParents(self, parents: list):
        self.parents = parents
    
    def testAgainstBest(self):
        pass 
    def llmCall(self): 
        pass
    def isWorking(self):
        pass 
    def evaluate(self): 
        if self.isFirstGen: 
            self.evaluation = self.llmCall()
    
        
