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
#from os import path
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.taxonomy.criteria import Criteria, SearchOperator,HasLoss, TypeOperator, OutputCriteria
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from src.taxonomy.create_taxonomy import *
from src.taxonomy.visualizeutils import visualizeTaxonomy






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



def llm_create_taxonomy(query : str) -> OutputCriteria:
    
    # constructing an example output criteria and search operator for the taxonomy
    op = SearchOperator(HasType=HasLoss )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    op2 = SearchOperator(Type=TypeOperator(name='layer_num_units'),Value=[600,3001],Op='range',Name='layer_num_units', HashOn='found' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])

    criteria1 = Criteria(Name='Has Loss Criteria')
    criteria1.add(op)

    criteria2 = Criteria(Name='Layer Num Units')
    criteria2.add(op2)

    op3 = SearchOperator(Op='cluster',Type=TypeOperator(Name='kmeans', Arguments=[4,'binary']), Value=['layer_num_units','dropout_rate'], HasType='hasLayer')
    criteria3 = Criteria(Name='KMeans Clustering')

    oc = OutputCriteria(criteriagroup=[criteria1,criteria2,criteria3], description="A taxonomy of loss at the top and a range of number of units and has kmeans on layer_num_units, dropout_rate, and and types of layers.").model_dump_json()
    
    # create a parser to parse output criteria
    parser = PydanticOutputParser(pydantic_object=OutputCriteria)

    criteriaprompt = ChatPromptTemplate([('system',system_prompt), ('human', '{user_input}')]).partial(format_instructions=parser.get_format_instructions())
    model = ChatOllama(model="deepseek-r1:32b-qwen-distill-q4_K_M",temperature=0.1)

    # a fixing parser if the original model doesnt work
    fixparser = OutputFixingParser.from_llm(parser=parser, llm=model)


    model.with_structured_output(OutputCriteria)
    chain = criteriaprompt | model

    #ontology_path = f"./data/owl/annett-o-test.owl" 
    ontology_path = f"./data/owl/fairannett-o.owl" 
    ontology = load_ontology(ontology_path=ontology_path)


    output = chain.invoke({'user_input': query,'oc': oc, 'schema': OutputCriteria.schema_json(indent=2)}).content
    output = re.sub(r"<think>.*?</think>\n?", "", output, flags=re.DOTALL)
   
    #print(output)
    #input('test')

    # fixing the criteria -- this may fail sometimes
    thecriteria = output = fixparser.parse(output)
    
    return thecriteria


if __name__ == '__main__':
    thecriteria = llm_create_taxonomy('What would you say is the taxonomy that preresents all neural network?')

    ontology_path = f"./data/owl/fairannett-o.owl" 
    ontology = load_ontology(ontology_path=ontology_path)
    
    taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    topnode, faceted, output = taxonomy_creator.create_taxonomy(format='graphml', faceted=True)
    
    with open('test.json', 'w') as handle:
        handle.write(json.dumps(serialize(faceted)))
    
