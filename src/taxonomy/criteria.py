from pydantic import BaseModel, ValidationError,Field
from typing import List, Optional, Dict, Literal


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


class TypeOperator(BaseModel):
    Name: str = Field("")
    Arguments: List[int|float|str] = Field([])

class SearchOperator(BaseModel):
    '''
    Name: SearchOperator
    Description: Has is for the edge properties like hasNetwork, hasLayer. Equals is for matching to specific names.
    '''
    HasType: Optional[str] = Field("")
    Type: Optional[TypeOperator] = None #Optional[str] = Field("")
    Name: Optional[str] = Field("")
    Op: Literal["cluster","sequal","none","range","scomp","leq","less","greater","geq"] = "none" #Optional[str] = Field("") 
    Value: Optional[str | int | float | list ] = Field("")
    HashOn: Optional[str] = Field("type")

    #has: Optional [ List ] = []
    #equals: Optional[ List ] = []

class Criteria(BaseModel):
    '''
    Name: Criteria
    Description: A class to contain the manual splitting criteria for the taxonomy.
    '''
    Searchs: List[SearchOperator] = Field([]) #= []
    
    Name: Optional[str] = Field("level")
    def add(self, operator:SearchOperator):
        self.Searchs.append(operator)

class OutputCriteria(BaseModel):
    """Always use this tool to structure your response to the user."""
    criteriagroup: List[Criteria] = Field(description='The levels of the taxonomy as written by the criteria in each element of this list.')
    description: str = Field(description="The description of the taxonomy created.")
