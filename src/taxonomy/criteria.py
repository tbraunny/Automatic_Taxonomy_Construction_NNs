from pydantic import BaseModel, ValidationError,Field, model_validator
from typing import List, Optional, Dict, Literal
from typing_extensions import Self

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
    Name: Literal["","kmeans","agg","graph"] = Field("", description="A field to define the type of clustering: supported is kmeans, agglomerative, graph clustering on networks, and no clustering. ") 
    Arguments: List[int|float|str] = []

    @model_validator(mode='after')
    def check_proper_cluster(self) -> Self:
        if (self.Name == 'kmeans' or self.Name == 'agg'or self.Name == 'graph') and not (len(self.Arguments) == 2 or len(self.Arguments) == 0): 
            raise ValueError("if name is kmeans or agg and arguments must have length of 2 or 0.")
        if (self.Name == 'kmeans' or self.Name == 'agg' or self.Name == 'graph') and len(self.Arguments) == 2 and type(self.Arguments[0])  != str and type(self.Arguments[1]) != str and not self.Arguments[0].numeric(): # enforcing
            raise ValueError("if name is kmeans or agg and arguments must be numeric for first value and second argument must be string.")
 
        return self

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
