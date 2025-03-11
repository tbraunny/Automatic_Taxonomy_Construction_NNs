from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict


AND = 'and'
OR = 'or'
LESS = 'less'
GREATeR = 'greater'
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
    HasType: Optional[str] = ""
    Type: Optional[str] = ""
    Name: Optional[str] = ""
    Op: Optional[str] = "" 
    Value: Optional[str | int | float | list ] = ""
    HashOn: Optional[str] = "type"
    #has: Optional [ List ] = []
    #equals: Optional[ List ] = []

class Criteria(BaseModel):
    '''
    Name: Criteria
    Description: A class to contain the manual splitting criteria for the taxonomy.
    '''
    criteria: List[SearchOperator] = []
    def add(self, operator:SearchOperator):
        self.criteria.append(operator)


        

