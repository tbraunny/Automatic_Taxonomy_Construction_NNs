import re
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import PydanticOutputParser
import requests
from bs4 import BeautifulSoup
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from graphviz import Source
import json
import re
import tiktoken
#from os import path
import os
from criteria import Criteria, SearchOperator,HasLoss
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from create_taxonomy import *
from visualizeutils import visualizeTaxonomy

class OutputCriteria(BaseModel):
    """Always use this tool to structure your response to the user."""
    criteriagroup: List[Criteria] = Field(description='The levels of the taxonomy as written by the criteria in each element of this list.')
    description: str = Field(description="The description of the taxonomy created.")

op = SearchOperator(HasType=HasLoss )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
op2 = SearchOperator(Type='layer_num_units',Value=[600,3001],Op='range',Name='layer_num_units', HashOn='found' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
#op = SearchOperator(has= [] , equals=[{'type':'name', 'value':'simple_classification_L2'}])
#op = SearchOperator(has= [] , equals=[{'type':'value','value':1000,'op':'greater','name':'layer_num_units'}])

criteria1 = Criteria(Name='Has Loss Criteria')
criteria1.add(op)

criteria2 = Criteria(Name='Layer Num Units')
criteria2.add(op2)

op3 = SearchOperator(Op='cluster',Type='kmeans(4,binary)', Value=['layer_num_units','dropout_rate'], HasType='hasLayer')
criteria3 = Criteria(Name='KMeans Clustering')



oc = OutputCriteria(criteriagroup=[criteria1,criteria2,criteria3], description="A taxonomy of loss at the top and a range of number of units and has kmeans on layer_num_units, dropout_rate, and and types of layers.").model_dump_json()
#print(criteria1.model_dump_json())





system_prompt = """ 
"You are an expert in constructing taxonomy-based queries using a structured data schema language. Your task is to define search constraints using SearchOperator objects and organize them into Criteria objects, ensuring that each layer is a union of elements.

When given a query about differentiating entities, such as classifying small vs. large neural networks, follow these guidelines:

Define the criteria where each criteria is a level in the taxonomy.
A criteria may have one or more SearchOperators.

Example Output:
To Create a taxonomy with the top layer representing loss and the bottom layer representing units between 600 and 3001:
{oc}


Supported HasTypes:

    Network Structure:
        hasNetwork
        hasLayer
        hasInputLayer
        hasOutputLayer
        hasRNNLayer

    Layer Properties:
        hasActivationFunction
        hasRegularizer
        hasWeights

    Training & Optimization:

        hasTrainingStrategy
        hasTrainingOptimizer
        hasTrainingSession
        hasTrainingStep
        hasPrimaryTrainingStep
        hasLoopTrainingStep
        hasPrimaryLoopTrainingStep
    Evaluation & Metrics:

        hasEvaluation
        hasMetric
        hasObjective
        hasCost
        hasLoss
        hasStopCondition
    
    Data & Labels:
        hasDataType
        hasLabels
        hasCharacteristicLabel
        hasTaskType
    
    Functionality & Characteristics:
        hasFunction
        hasTrainedModel
        hasInitialState

Supported Ops -- these must be specified in the op field of the search operator:
    less
    greater
    leq
    geq
    equal
    scomp
    range

Please format your answer as given by the example above or:
{oc}
"""


system_prompt = '''
You are an expert in constructing taxonomy-based queries using a structured data schema language. Your primary goal is to generate search constraints using SearchOperator objects and arrange them into Criteria objects. Each Criteria corresponds to a single layer in the taxonomy, and each layer (i.e., each Criteria) is a union of elements.

When you receive a request—such as classifying small vs. large neural networks or creating a taxonomy with a top layer representing a loss function and a bottom layer representing a range of units—follow these guidelines:

1. Taxonomy Construction
   - Each Criteria represents one level in the taxonomy.
   - A single level (one Criteria) can contain one or more SearchOperator objects.
   - The final output should be a list of Criteria objects arranged in hierarchical order.

2. SearchOperator Definition
   - Use the HasType field to specify the ontology property to filter on (e.g., hasLayer, hasEvaluation).
   - Use the HasType to search across all has types, but note that it doesn't support the Op field.
   - Use the Op field to specify the comparison operator. Supported operators are: cluster, less, greater, leq, geq, equal, scomp, and range.
   - With the cluster Op the Type field is used to specify the type of clustering. The only supported clustering is kmeans. Example: Type: 'kmeans(4,binary)' -- kmeans with four clusters and encoding words to binary. It must be specified this way. The binary option is only supported at this time. Specify what values to cluster on in the Value field as list of values and a single type can be specified in the HasType field for a type which both values and types can be clustered on.
   - Use the Value field to define the target value or range.

3. Supported HasType Values:
   - Network Structure:
     - hasNetwork
     - hasLayer
     - hasInputLayer
     - hasOutputLayer
     - hasRNNLayer
   - Layer Properties:
     - hasActivationFunction
     - hasRegularizer
     - hasWeights
   - Training & Optimization:
     - hasTrainingStrategy
     - hasTrainingOptimizer
     - hasTrainingSession
     - hasTrainingStep
     - hasPrimaryTrainingStep
     - hasLoopTrainingStep
     - hasPrimaryLoopTrainingStep
   - Evaluation & Metrics:
     - hasEvaluation
     - hasMetric
     - hasObjective
     - hasCost
     - hasLoss
     - hasStopCondition
   - Data & Labels:
     - hasDataType
     - hasLabels
     - hasCharacteristicLabel
     - hasTaskType
   - Functionality & Characteristics:
     - hasFunction
     - hasTrainedModel
     - hasInitialState

4. Response Format
   - Return only the structured DSL in Python (or JSON) format.
   - Format your answer as a Python code block similar to the example below.
   - Do not include additional explanation unless explicitly requested.

Example Output:
To create a taxonomy with the top layer representing loss and the bottom layer representing units in the range 600 to 3001, you might output:

-----------------------------------------------------------
{oc}
-----------------------------------------------------------


Replace the example properties and values as needed based on the specific query. Your output must be a structured list of Criteria objects using the defined DSL.
Return only the format with no ``` ```
'''


parser = PydanticOutputParser(pydantic_object=OutputCriteria)

criteriaprompt = ChatPromptTemplate([('system',system_prompt), ('human', '{user_input}')]).partial(format_instructions=parser.get_format_instructions())
print(criteriaprompt)
#conservative_model = OllamaLLM(model="qwq:32b",temperature=0)
#conservative_model = OllamaLLM(model="llama3.2",temperature=0)
#conservative_model = ChatOllama(model="neuralexpert:latest",temperature=0)
#conservative_model = ChatOllama(model="qwq:32b",temperature=0)
llama_model = ChatOllama(model="llama3.2",temperature=0.1)
conservative_model = ChatOllama(model="deepseek-r1:32b-qwen-distill-q4_K_M",temperature=0.1)
#conservative_model = ChatOllama(model="neuralexpert",temperature=0.2)

fixparser = OutputFixingParser.from_llm(parser=parser, llm=conservative_model)


#if not streaming:
#    conservative_model = conservative_model.bind_tools([OutputCriteria])


conservative_model.with_structured_output(OutputCriteria)
chain = criteriaprompt | conservative_model

#ontology_path = f"./data/owl/annett-o-test.owl" 
ontology_path = f"./data/owl/annett-o.owl" 
ontology = load_ontology(ontology_path=ontology_path)


#output = chain.invoke({'user_input': 'Give me a taxonomy that splits small networks use the range operator.','oc': oc}).content
output = chain.invoke({'user_input': 'Give me a taxonomy that categorizes based on types of layers, optimizer, and loss function and uses kmeans.','oc': oc, 'schema': OutputCriteria.schema_json(indent=2)}).content
output = re.sub(r"<think>.*?</think>\n?", "", output, flags=re.DOTALL)
thecriteria = output = fixparser.parse(output)
print(type(output))
print(thecriteria)
input()
taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
topnode, faceted, output = taxonomy_creator.create_taxonomy(format='graphml', faceted=True)

print(output)
visualizeTaxonomy(output)
print(thecriteria)
print(faceted)

with open('test.json', 'w') as handle:
    handle.write(json.dumps(serialize(faceted)))

'''if not streaming:
    output = chain.invoke({'user_input': 'Construct a taxonomy that splits on 10 to 100 neurons.','oc': oc})
    #print(output)
    if len(output.tool_calls):
        print('never gonna give you up :). It was successful')
        print(output.tool_calls[0]['args'])
        thecriteria = OutputCriteria(**output.tool_calls[0]['args'])



    taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    print(taxonomy_creator.create_taxonomy(format='graphml'))
    print(thecriteria)
'''
