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

from utils.llm_service import load_environment_llm




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
   - Use the Op field to specify the comparison operator to sue clustering. Supported operators are: cluster and none    
   - With the cluster Op the Type field is used to specify the type of clustering. The only supported clustering is kmeans. kmeans with four clusters and encoding words to binary. It must be specified this way. The binary option is only supported at this time. Specify what values to cluster on in the Value field as list of values and a single type can be specified in the HasType field for a type which both values and types can be clustered on. Here is the spec: {typeoperator}
   - Use the Value field to define what values you want to query against.
   - The Value field takes a list of ValueOperator with schema within the Criteria: {valueoperator}
   - The Value field has the following supported ops: less, greater, leq, geq, equal, scomp, and range, name, has. name is used to query for specific names and has for querying specific types. The has can be used to query for things like hasLayer, hasEvaluation.

3. Supported 'has' op Values:
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
----------

Replace the example properties and values as needed based on the specific query. Your output must be a structured list of Criteria objects using the defined DSL.
Return only the format with no ``` ```
'''

# system_prompt = '''
# You are a knight of the order of the unicorn and are an expert in constructing taxonomy-based queries using a structured data schema language. Your primary goal is to generate search constraints using SearchOperator objects and arrange them into Criteria objects. Each Criteria corresponds to a single layer in the taxonomy, and each layer (i.e., each Criteria) is a union of elements.

# When you receive a request—such as classifying small vs. large neural networks or creating a taxonomy with a top layer representing a loss function and a bottom layer representing a range of units—follow these guidelines:

# 1. Taxonomy Construction
#    - Each Criteria represents one level in the taxonomy.
#    - A single level (one Criteria) can contain one or more SearchOperator objects.
#    - The final output should be a list of Criteria objects arranged in hierarchical order.
# 2. SearchOperator Definition
#    - The name for Search operator does not change taxonomy and it is recommended to make it something descriptive
#    - Use the Cluster field to specify the comparison operator to use clustering. Supported operators are: cluster and none    
#    - The Type field is used to specify the type of clustering. The only supported clustering is kmeans. kmeans with four clusters and encoding words to binary. It must be specified this way. The binary option is only supported at this time. Specify what values to cluster on in the Value field as list of values and a single type can be specified in the HasType field for a type which both values and types can be clustered on. Here is the spec: {typeoperator}
#    - Use the Value field to define what values you want to query against and is Value Operator.
# 3. ValueOperator Definition
#    - The Value field takes a list of int,string,float.
#    - The Op field has the following supported ops: less, greater, leq, geq, equal, scomp, and range, name, has. name is used to query for specific names and has for querying specific types. The has can be used to query for things like hasLayer, hasEvaluation.
#    - The Name field has the following supported names: {properties}
# 4. Response Format
#    - Return only the structured DSL in Python (or JSON) format.
#    - Format your answer as a Python code block similar to the example below or following the schema below.
#    - Do not include additional explanation unless explicitly requested.

# Example API block: {oc}
# Do your best and follow the following schema no matter what: {schema}
# '''

# system_prompt = '''
# You are an expert system for constructing machine-readable taxonomy queries from natural language prompts. Your task is to convert user queries into structured search constraints using a Python-based DSL composed of `SearchOperator`, `ValueOperator`, `TypeOperator`, and `Criteria` objects. These form a hierarchy of constraints representing facets and layers of a taxonomy.

# Guidelines:
# 1. **Taxonomy Layers**:
#    - Each `Criteria` represents a level in the taxonomy.
#    - Each level (Criteria) includes one or more `SearchOperator` objects.

# 2. **SearchOperator Definition**:
#    - `Name`: A descriptive label for the facet.
#    - `Cluster`: Accepts `"cluster"` or `"none"`. Use `"cluster"` for grouping similar items via clustering.
#    - `Type`: Used only when `Cluster='cluster'`. Supported: `kmeans`.
#        - For kmeans, use: `TypeOperator(Name='kmeans', Arguments=['4','binary'])`.
#    - `Value`: A list of `ValueOperator` defining the values or ranges of interest.
#    - `HashOn`: Determines what to hash on: "type", "found", "name", or "value".

# 3. **ValueOperator Definition**:
#    - `Name`: The ontology property you want to filter on (e.g., `layer_num_units`, `hasLoss`).
#    - `Op`: The comparison type. Supported ops: `less`, `greater`, `leq`, `geq`, `range`, `equal`, `sequal`, `scomp`, `none`, `name`, `has`.
#    - `Value`: A list of values (int, float, or string).

# 4. **Ontology Property Suggestions**:
#    - **Network Structure**: hasNetwork, hasLayer, hasInputLayer, hasOutputLayer, hasRNNLayer
#    - **Layer Properties**: hasActivationFunction, hasRegularizer, hasWeights
#    - **Training & Optimization**: hasTrainingStrategy, hasTrainingOptimizer, hasTrainingSession, hasTrainingStep
#    - **Evaluation & Metrics**: hasEvaluation, hasMetric, hasObjective, hasCost, hasLoss
#    - **Data & Labels**: hasDataType, hasLabels, hasTaskType
#    - **Functionality**: hasFunction, hasTrainedModel

# 5. **Output Format**:
#    - Return a Python object that instantiates `OutputCriteria(criteriagroup=[...], description=...)`.
#    - Do NOT include explanation text or markdown. Just the Python object definition.

# **Example Prompt**: 
# "Classify neural networks by loss function, then cluster by layer_num_units and dropout_rate."

# **Expected Output**:
# OutputCriteria(
#   criteriagroup=[
#     Criteria(Name='Loss Split', Searchs=[
#       SearchOperator(Name='Loss Type', Cluster='none', Value=[
#         ValueOperator(Name='hasLoss', Op='has', Value=[])
#       ])
#     ]),
#     Criteria(Name='KMeans Cluster by Layer Info', Searchs=[
#       SearchOperator(Name='Cluster Layers', Cluster='cluster', Type=TypeOperator(Name='kmeans', Arguments=['4','binary']),
#         Value=[
#           ValueOperator(Name='layer_num_units', Op='name'),
#           ValueOperator(Name='dropout_rate', Op='name')
#         ]
#       )
#     ])
#   ],
#   description='A taxonomy based on loss function, followed by clustering on layer units and dropout rate.'
# )
# '''


def llm_create_taxonomy(query : str, ontology) -> OutputCriteria:
    
    # constructing an example output criteria and search operator for the taxonomy
    op = SearchOperator(Value=[ValueOperator(Name='HasLoss',Op='has')])#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    op2 = SearchOperator(Value=[ValueOperator(Name='layer_num_units',Value=[600,3001])],Cluster='none',Name='layer_num_units', HashOn='found' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])

    criteria1 = Criteria(Name='Has Loss Criteria')
    criteria1.add(op)

    criteria2 = Criteria(Name='Layer Num Units')
    criteria2.add(op2)

    op3 = SearchOperator(Cluster='cluster',Type=TypeOperator(Name='kmeans', Arguments=['4','binary']), Value=[ValueOperator(Name='layer_num_units',Op='name'),ValueOperator(Name='dropout_rate',Op='name')])
    criteria3 = Criteria(Name='KMeans Clustering')

    oc = OutputCriteria(criteriagroup=[criteria1,criteria2,criteria3], description="A taxonomy of loss at the top and a range of number of units and has kmeans on layer_num_units, dropout_rate, and and types of layers.").model_dump_json()
    
    #model='qwq:32b'
    model = 'llama3.2:latest'
    model = ChatOllama(model=model,temperature=0.1, top_p= 1, repeat_penalty=1, num_ctx=5000)
    # keep te above just in case things dont't work

    llm = load_environment_llm(temperature=0.1,num_ctx=5000)
    model = llm.llm

    # a fixing parser if the original model doesnt work

    # create a parser to parse output criteria
    parser = PydanticOutputParser(pydantic_object=OutputCriteria)
    criteriaprompt = ChatPromptTemplate([('system',system_prompt), ('human', '{user_input}')]).partial(format_instructions=parser.get_format_instructions())
    fixparser = OutputFixingParser.from_llm(parser=parser, llm=model)

    #llm.
    #llm.llm.with_structured_output(OutputCriteria)
    chain = criteriaprompt | model

    #ontology_path = f"./data/owl/annett-o-test.owl" 
    #ontology_path = f"./data/owl/fairannett-o.owl" 
    #ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    #for prop in ontology.get_properties():
    properties = list(ontology.data_properties()) + list(ontology.object_properties()) + list(ontology.classes())
    properties = [prop.name for prop in properties]
    print(properties)

    output = chain.invoke({'user_input': query,'oc': oc,'schema': json.dumps(OutputCriteria.model_json_schema()), 'valueoperator': json.dumps(ValueOperator.model_json_schema()), 'typeoperator': json.dumps(TypeOperator.model_json_schema()), "properties": json.dumps(properties).replace('"','') } ) #.content
    
    print(output)

    output = output.content
    output = re.sub(r"<think>.*?</think>\n?", "", output, flags=re.DOTALL)


    # fixing the criteria -- this may fail sometimes
    thecriteria = output = fixparser.parse(output)
    handle = open('criteria.json','w')
    handle.write(json.dumps(thecriteria.model_dump()))
    handle.close()

    return thecriteria


if __name__ == '__main__':
    # ontology_path = f"./data/owl/fairannett-o.owl" 
    ontology_path = f"./data/Annett-o/annett-o-0.1.owl"
    ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    thecriteria = llm_create_taxonomy('What would you say is the taxonomy that represents all neural network?', ontology)
    print(f"THE_CRITERIA: {thecriteria}")
    #  logging.getLogger().setLevel(logging.WARNING)
    # logger.info("Loading ontology.")
    # ontology_path = f"./data/Annett-o/annett-o-0.1.owl"

    # # Example Criteria...
    # op2 = SearchOperator(HashOn="value",Value=[ValueOperator(Name='hasLayer',Op="has")],Cluster='none',Name='layer_num_units' )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    # op = SearchOperator(Cluster="cluster", Value=[ValueOperator(Name="layer_num_units",Value=[10],Op="less")],Name='layer_num_units', HashOn='found', Type=TypeOperator(Name="kmeans") )#, equals=[{'type':'name', 'value':'simple_classification_L2'}])
    
    # criteria = Criteria(Name='Layer Num Units')
    # criteria.add(op)
    # # #criteria.add(op2)

    # criterias = [criteria]
    # ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    # logger.info(ontology.instances)
    # logger.info("Ontology loaded.")
    # logger.info("Creating taxonomy from Annetto annotations.")

    # # taxonomy creator
    taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    format='json'
    topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(format=format,faceted=True)

    # # create a dataframe
    df = create_tabular_view_from_faceted_taxonomy(taxonomy_str=json.dumps(serialize(facetedTaxonomy)), format="json")
    df.to_csv("./test_tax_yuh.csv")
    import dataframe_image as dfi
    dfi.export(df, "./test_tax_yuh.png")
    # taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    # topnode, faceted, output = taxonomy_creator.create_taxonomy(format='json', faceted=True)
    
    # with open('test.json', 'w') as handle:
    #     handle.write(json.dumps(serialize(faceted)))
    
