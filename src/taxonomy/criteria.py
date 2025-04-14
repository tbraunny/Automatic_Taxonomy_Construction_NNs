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
    '''
    Name: TypeOperator
    Description: Is used to specify types and is mainly used with clustering.
    '''
    Name: str = ""
    Arguments: List[int|float|str] = []

class ValueOperator(BaseModel):
    '''
    Name: ValueOperator
    Description: Is used to query for things in a ontology and has a number of different operators.
    Name: Is a property of the knowledge base.
    '''
    Name: str = "" #Field("", description="A property of the knowledgebase that the values represent.")
    Op: Literal["sequal","none","range","scomp","leq","less","greater","geq","name",'has'] = "none"
    Value: List[str | int | float ] = []

class SearchOperator(BaseModel):
    '''
    Name: SearchOperator
    Description: Has is for the edge properties like hasNetwork, hasLayer. Equals is for matching to specific names.
    '''
    #HasType: Optional[str] = Field("")
    Type: Optional[TypeOperator] = None #Optional[str] = Field("")
    Name: str = Field("")
    Cluster: Literal["cluster","none"] = "none" #Optional[str] = Field("") 
    Value: List[ValueOperator] = []
    HashOn: Literal["type", "found"] = "type"

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
