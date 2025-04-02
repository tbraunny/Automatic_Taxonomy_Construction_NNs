from pydantic import BaseModel, ValidationError,Field
from typing import List, Optional, Dict


AND = 'and'
OR = 'or'
LESS = 'less'
GREATER = 'greater'
LESS_EQUAL = 'leq'
GREATER_EQUAL = 'geq'
EQUAL = 'equal'
SCOMP = 'scomp'
RANGE = 'range'
HasTaskType = 'hasTaskType'
HasLayer = 'hasLayer'
HasLoss = 'hasLoss'

class SearchOperator(BaseModel):
    '''
    Name: SearchOperator
    Description: Has is for the edge properties like hasNetwork, hasLayer. Equals is for matching to specific names.
    '''
    HasType: Optional[str] = Field("")
    Type: Optional[str] = Field("")
    Name: Optional[str] = Field("")
    Op: Optional[str] = Field("") 
    Value: Optional[str | int | float | list ] = Field("")
    HashOn: Optional[str] = Field("type")

    #has: Optional [ List ] = []
    #equals: Optional[ List ] = []

class Criteria(BaseModel):
    '''
    Name: Criteria
    Description: A class to contain the manual splitting criteria for the taxonomy.
    '''
    Searchs: List[SearchOperator] = [] #= []
    
    Name: Optional[str] = Field("level")
    def add(self, operator:SearchOperator):
        self.Searchs.append(operator)


        

