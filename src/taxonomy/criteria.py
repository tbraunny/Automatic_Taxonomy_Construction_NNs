from pydantic import BaseModel, ValidationError
from typing import List, Optional, Dict


AND = 'and'
OR = 'or'
LESS = '<'
GREATOR = '>'
LESS_EQUAL = '<='
GREATER_EQUAL = '>='
EQUAL = '=='

HasTaskType = 'hasTaskType'
HasLayer = 'hasLayer'
HasLoss = 'hasLoss'
class SearchOperator(BaseModel):
    '''
    Name: SearchOperator
    Description: Has is for the edge properties like hasNetwork, hasLayer. Equals is for matching to specific names.
    '''
    has: Optional [ List ] = []
    equals: Optional[ List ] = []

class Criteria(BaseModel):
    '''
    Name: Criteria
    Description: A class to contain the manual splitting criteria for the taxonomy.
    '''
    criteria: List[SearchOperator] = []
    def add(self, operator:SearchOperator):
        self.criteria.append(operator)


        

