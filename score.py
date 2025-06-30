import time
import os 
from .worker import agent


class score: 
    def __init__(self, Agent: agent, 
                 llm_client, 
                 evaluation_prompt: str):
        
        self.Agent = agent
        self.llm_client = llm_client
        self.evaluation_prompt = evaluation_prompt

    def evaluate(self):
        pass
