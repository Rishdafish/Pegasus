from .manager import Manager 
from .prompts import prompts 
class agent(): 
    def __init__(self, role, task, isFirstGen = False, id: str = None, code = None, parents = []): 
        super().__init__()
        self.role = role
        self.task = task
        self.isEvaluated = False
        self.parents = parents    
        self.isFirstGen = isFirstGen
        self.score = 0 
        self.id = id
        self.code = code


    def setParents(self, parents: list):
        self.parents = parents
    def setScore(self, score: int): 
        self.score = score
    def getScore(self):
        return self.score
    def getId(self):
        return self.id
    def getParents(self):
        return self.parents

    def createChildren(self, parent, numChildren) -> tuple:
        pass
    
    def llmCall(self): 
        pass
    
        
