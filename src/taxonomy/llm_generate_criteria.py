import re
from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import PydanticOutputParser
import requests
from bs4 import BeautifulSoup
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
import json
import re
import random

# from os import path
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.taxonomy.criteria import (
    Criteria,
    SearchOperator,
    HasLoss,
    TypeOperator,
    OutputCriteria,
)
from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from src.taxonomy.create_taxonomy import *
from src.taxonomy.visualizeutils import visualizeTaxonomy

from utils.llm_service import load_environment_llm


# system_prompt = """ 
# "You are an expert in constructing taxonomy-based queries using a structured data schema language. Your task is to define search constraints using SearchOperator objects and organize them into Criteria objects, ensuring that each layer is a union of elements.

# When given a query about differentiating entities, such as classifying small vs. large neural networks, follow these guidelines:

# Define the criteria where each criteria is a level in the taxonomy.
# A criteria may have one or more SearchOperators.

# Example Output:
# To Create a taxonomy with the top layer representing loss and the bottom layer representing units between 600 and 3001:
# {oc}


# Supported HasTypes:

#     Network Structure:
#         hasNetwork
#         hasLayer
#         hasInputLayer
#         hasOutputLayer
#         hasRNNLayer

#     Layer Properties:
#         hasActivationFunction
#         hasRegularizer
#         hasWeights

#     Training & Optimization:

#         hasTrainingStrategy
#         hasTrainingOptimizer
#         hasTrainingSession
#         hasTrainingStep
#         hasPrimaryTrainingStep
#         hasLoopTrainingStep
#         hasPrimaryLoopTrainingStep
#     Evaluation & Metrics:

#         hasEvaluation
#         hasMetric
#         hasObjective
#         hasCost
#         hasLoss
#         hasStopCondition
    
#     Data & Labels:
#         hasDataType
#         hasLabels
#         hasCharacteristicLabel
#         hasTaskType
    
#     Functionality & Characteristics:
#         hasFunction
#         hasTrainedModel
#         hasInitialState

# Supported Ops -- these must be specified in the op field of the search operator:
#     less
#     greater
#     leq
#     geq
#     equal
#     scomp
#     range

# Please format your answer as given by the example above or:
# {oc}
# """


# system_prompt = """
# You are an expert in constructing taxonomy-based queries using a structured data schema language. Your primary goal is to generate search constraints using SearchOperator objects and arrange them into Criteria objects. Each Criteria corresponds to a single layer in the taxonomy, and each layer (i.e., each Criteria) is a union of elements.

# When you receive a request—such as classifying small vs. large neural networks or creating a taxonomy with a top layer representing a loss function and a bottom layer representing a range of units—follow these guidelines:

# 1. Taxonomy Construction
#    - Each Criteria represents one level in the taxonomy.
#    - A single level (one Criteria) can contain one or more SearchOperator objects.
#    - The final output should be a list of Criteria objects arranged in hierarchical order.

# 2. SearchOperator Definition
#    - Use the Op field to specify the comparison operator to sue clustering. Supported operators are: cluster and none    
#    - With the cluster Op the Type field is used to specify the type of clustering. The only supported clustering is kmeans. kmeans with four clusters and encoding words to binary. It must be specified this way. The binary option is only supported at this time. Specify what values to cluster on in the Value field as list of values and a single type can be specified in the HasType field for a type which both values and types can be clustered on. Here is the spec: {typeoperator}
#    - Use the Value field to define what values you want to query against.
#    - The Value field takes a list of ValueOperator with schema within the Criteria: {valueoperator}
#    - The Value field has the following supported ops: less, greater, leq, geq, equal, scomp, and range, name, has. name is used to query for specific names and has for querying specific types. The has can be used to query for things like hasLayer, hasEvaluation.

# 3. Supported 'has' op Values:
#    - Network Structure:
#      - hasNetwork
#      - hasLayer
#      - hasInputLayer
#      - hasOutputLayer
#      - hasRNNLayer
#    - Layer Properties:
#      - hasActivationFunction
#      - hasRegularizer
#      - hasWeights
#    - Training & Optimization:
#      - hasTrainingStrategy
#      - hasTrainingOptimizer
#      - hasTrainingSession
#      - hasTrainingStep
#      - hasPrimaryTrainingStep
#      - hasLoopTrainingStep
#      - hasPrimaryLoopTrainingStep
#    - Evaluation & Metrics:
#      - hasEvaluation
#      - hasMetric
#      - hasObjective
#      - hasCost
#      - hasLoss
#      - hasStopCondition
#    - Data & Labels:
#      - hasDataType
#      - hasLabels
#      - hasCharacteristicLabel
#      - hasTaskType
#    - Functionality & Characteristics:
#      - hasFunction
#      - hasTrainedModel
#      - hasInitialState

# 4. Response Format
#    - Return only the structured DSL in Python (or JSON) format.
#    - Format your answer as a Python code block similar to the example below.
#    - Do not include additional explanation unless explicitly requested.

# Example Output:
# To create a taxonomy with the top layer representing loss and the bottom layer representing units in the range 600 to 3001, you might output:

# -----------------------------------------------------------
# {oc}
# ----------

# Replace the example properties and values as needed based on the specific query. Your output must be a structured list of Criteria objects using the defined DSL.
# Return only the format with no ``` ```
# """

# system_prompt = """
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
# """

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

system_prompt = """
You are a world-class expert at designing taxonomies for machine learning model ontologies.

Your task is to **automatically generate a structured, multi-layer taxonomy** that separates models based on:
- Specific layer types (e.g., Dense, Conv2D, BatchNormalization)
- Specific task types (e.g., classification, regression, unsupervised)
- Clear numerical ranges (e.g., batch size 32–128, units > 512)
- Clustering across multiple features (e.g., layer units + dropout rates)
- Presence or absence of architectural or optimization features (e.g., Dropout layers, BatchNormalization, EarlyStopping)

---

### RULES:

1. **Criteria Layering**:
   - Each Criteria represents **one layer** in the taxonomy split.
   - Each Criteria contains one or more SearchOperator objects.

2. **SearchOperator Definition**:
   - Cluster field can be `"none"`, `"cluster"`, or `"agg"`.
   - If Cluster is `"cluster"`, use TypeOperator(Name="kmeans", Arguments=["4", "binary"]).
   - If Cluster is `"agg"`, use TypeOperator(Name="agg", Arguments=["3", "binary"]).
   - If Cluster is `"none"`, omit the Type field.
   - The Value field contains one or more ValueOperator objects.

3. **ValueOperator Definition**:
   - Supported Ops: `"less"`, `"greater"`, `"leq"`, `"geq"`, `"equal"`, `"sequal"`, `"scomp"`, `"range"`, `"name"`, `"has"`.
   - Value can be int, float, or string depending on property type.

4. **Preferred Construction Strategy**:
   - First split by **Task Type** (e.g., supervised vs. unsupervised).
   - Then by **Layer Types** (e.g., presence of Conv2D, AttentionLayer, Residuals).
   - Then by **Hyperparameters** (e.g., units, dropout, batch size, learning rate).
   - Use **Value Ranges** whenever possible before clustering.
   - Only apply **Clustering (kmeans or agg)** for fine-grained splits when scalar splits are insufficient.

5. **Ontology Fields**:
   - Only use field names provided here: {properties}.

6. **Output Format**:
   - Return a valid `OutputCriteria` object.
   - Do not include any markdown formatting, explanations, or free text.
   - Match the schema exactly: {schema}.

---
### RESPONSE FORMAT:

Return a valid `OutputCriteria` object containing a list of `Criteria`.

Do not include any additional text or markdown formatting.

Follow the schema:
{schema}

---

### EXAMPLES:

Example Api Block:

{oc}
"""

def llm_create_taxonomy(query: str, ontology) -> OutputCriteria:

    # constructing an example output criteria and search operator for the taxonomy
    # --- Example Criteria Definitions ---
    criteria_list = []

    # Presence Check (has)
    criteria_has_dropout = Criteria(Name="Has Dropout Layer")
    criteria_has_dropout.add(
        SearchOperator(
            Name="hasLayer",
            Cluster="none",
            Value=[ValueOperator(Name="hasLayer", Op="has", Value=["Dropout"])],
            HashOn="name",
        )
    )
    criteria_list.append(criteria_has_dropout)

    # Exact String Match (name)
    criteria_layer_type = Criteria(Name="Layer Type Match")
    criteria_layer_type.add(
        SearchOperator(
            Name="layer_type",
            Cluster="none",
            Value=[
                ValueOperator(Name="layer_type", Op="name", Value=["Dense", "Conv2D"])
            ],
            HashOn="name",
        )
    )
    criteria_list.append(criteria_layer_type)

    # Value Range (range)
    criteria_num_units = Criteria(Name="Layer Units Range")
    criteria_num_units.add(
        SearchOperator(
            Name="layer_num_units",
            Cluster="none",
            Value=[ValueOperator(Name="layer_num_units", Op="range", Value=[64, 1024])],
            HashOn="found",
        )
    )
    criteria_list.append(criteria_num_units)

    # Less Than
    criteria_dropout_small = Criteria(Name="Dropout Rate < 0.2")
    criteria_dropout_small.add(
        SearchOperator(
            Name="dropout_rate",
            Cluster="none",
            Value=[ValueOperator(Name="dropout_rate", Op="less", Value=[0.2])],
            HashOn="value",
        )
    )
    criteria_list.append(criteria_dropout_small)

    # Greater Than
    criteria_large_lr = Criteria(Name="Learning Rate > 0.01")
    criteria_large_lr.add(
        SearchOperator(
            Name="learning_rate",
            Cluster="none",
            Value=[ValueOperator(Name="learning_rate", Op="greater", Value=[0.01])],
            HashOn="value",
        )
    )
    criteria_list.append(criteria_large_lr)

    # Greater Equal
    criteria_big_batch = Criteria(Name="Batch Size >= 64")
    criteria_big_batch.add(
        SearchOperator(
            Name="batch_size",
            Cluster="none",
            Value=[ValueOperator(Name="batch_size", Op="geq", Value=[64])],
            HashOn="value",
        )
    )
    criteria_list.append(criteria_big_batch)

    # Less Equal
    criteria_epochs = Criteria(Name="Epochs <= 100")
    criteria_epochs.add(
        SearchOperator(
            Name="epochs",
            Cluster="none",
            Value=[ValueOperator(Name="epochs", Op="leq", Value=[100])],
            HashOn="value",
        )
    )
    criteria_list.append(criteria_epochs)

    # Strict Equality (sequal)
    criteria_kernel_size = Criteria(Name="Kernel Size = 3")
    criteria_kernel_size.add(
        SearchOperator(
            Name="conv_kernel_size",
            Cluster="none",
            Value=[ValueOperator(Name="conv_kernel_size", Op="sequal", Value=[3])],
            HashOn="value",
        )
    )
    criteria_list.append(criteria_kernel_size)

    # String Comparison (scomp)
    criteria_optimizer = Criteria(Name="Optimizer is Adam-like")
    criteria_optimizer.add(
        SearchOperator(
            Name="optimizer",
            Cluster="none",
            Value=[ValueOperator(Name="optimizer", Op="scomp", Value=["adam"])],
            HashOn="name",
        )
    )
    criteria_list.append(criteria_optimizer)

    # --- Clustered Criteria ---

    # KMeans Cluster on layer size + dropout
    criteria_kmeans = Criteria(Name="KMeans Cluster: Units + Dropout")
    criteria_kmeans.add(
        SearchOperator(
            Name="cluster_kmeans",
            Cluster="cluster",
            Type=TypeOperator(Name="kmeans", Arguments=["4", "binary"]),
            Value=[
                ValueOperator(Name="layer_num_units", Op="name"),
                ValueOperator(Name="dropout_rate", Op="name"),
            ],
            HashOn="type",
        )
    )
    criteria_list.append(criteria_kmeans)

    # Agglomerative Cluster on optimizer and loss
    criteria_agg = Criteria(Name="Agglomerative Cluster: Optimizer + Loss")
    criteria_agg.add(
        SearchOperator(
            Name="cluster_agg",
            Cluster="cluster",
            Type=TypeOperator(Name="agg", Arguments=["3", "binary"]),
            Value=[
                ValueOperator(Name="optimizer", Op="name"),
                ValueOperator(Name="hasLoss", Op="name"),
            ],
            HashOn="type",
        )
    )
    criteria_list.append(criteria_agg)

    # Graph-based clustering on topology + layers
    criteria_graph = Criteria(Name="Graph Cluster: Layers & Structure")
    criteria_graph.add(
        SearchOperator(
            Name="cluster_graph",
            Cluster="cluster",
            Type=TypeOperator(Name="graph", Arguments=["5", "binary"]),
            Value=[
                ValueOperator(Name="hasLayer", Op="name"),
                ValueOperator(Name="hasStructure", Op="name"),
            ],
            HashOn="type",
        )
    )
    criteria_list.append(criteria_graph)
    ex_description = """
    This taxonomy separates neural network models into distinct groups based on structural properties, hyperparameters, and configurations. It uses a combination of direct value-based splits (such as the presence of specific layers or ranges of hyperparameter values) and cluster-based splits (such as k-means, agglomerative, and graph clustering on layer counts, dropout rates, or network topology). The taxonomy applies comparison operators (e.g., less than, greater than, equal to, range matching, string comparison) to properties like layer types, number of units, loss functions, dropout rates, learning rates, optimizers, and more. Clustered splits are used to group models with similar quantitative characteristics when direct value matching is insufficient. Overall, the taxonomy aims to create a structured, multi-level organization that distinguishes models by both explicit attributes and latent clusters based on model architecture or hyperparameter similarities.
    """
    test_criterias = []
    # --- Scalar Filters with Fine-Grained Ranges ---
    test_criterias.append(Criteria(
        Name="Dropout Rate Between 0.2 and 0.5",
        Searchs=[SearchOperator(
            Name="dropout_rate",
            Value=[ValueOperator(Name="dropout_rate", Op="range", Value=[0.2, 0.5])],
            HashOn="value"
        )]
    ))

    test_criterias.append(Criteria(
        Name="Learning Rate < 0.001",
        Searchs=[SearchOperator(
            Name="learning_rate",
            Value=[ValueOperator(Name="learning_rate", Op="less", Value=[0.001])],
            HashOn="value"
        )]
    ))

    # --- Activation Function (Exact Match & Presence) ---
    test_criterias.append(Criteria(
        Name="Has ReLU Activation",
        Searchs=[SearchOperator(
            Name="hasActivationFunction",
            Value=[ValueOperator(Name="hasActivationFunction", Op="sequal", Value=["ReLU"])],
            HashOn="name"
        )]
    ))

    test_criterias.append(Criteria(
        Name="Missing Activation Function",
        Searchs=[SearchOperator(
            Name="hasActivationFunction",
            Value=[ValueOperator(Name="hasActivationFunction", Op="none")],
            HashOn="found"
        )]
    ))

    # --- Regularization and Weight Decay ---
    test_criterias.append(Criteria(
        Name="L2 Regularization > 0.0001",
        Searchs=[SearchOperator(
            Name="hasRegularizer",
            Value=[ValueOperator(Name="hasRegularizer", Op="greater", Value=[0.0001])],
            HashOn="value"
        )]
    ))

    # --- Strategy Types (e.g., Optimization/Training) ---
    test_criterias.append(Criteria(
        Name="Uses Early Stopping",
        Searchs=[SearchOperator(
            Name="hasTrainingStrategy",
            Value=[ValueOperator(Name="hasTrainingStrategy", Op="sequal", Value=["EarlyStopping"])],
            HashOn="name"
        )]
    ))

    # --- Task Types ---
    test_criterias.append(Criteria(
        Name="Unsupervised Learning Tasks",
        Searchs=[SearchOperator(
            Name="hasTaskType",
            Value=[ValueOperator(Name="hasTaskType", Op="sequal", Value=["UnsupervisedLearning"])],
            HashOn="name"
        )]
    ))

    # --- Layer Types (Architectural Specificity) ---
    test_criterias.append(Criteria(
        Name="Includes Attention Layer",
        Searchs=[SearchOperator(
            Name="hasLayer",
            Value=[ValueOperator(Name="hasLayer", Op="sequal", Value=["AttentionLayer"])],
            HashOn="name"
        )]
    ))

    test_criterias.append(Criteria(
        Name="Includes Residual Connections",
        Searchs=[SearchOperator(
            Name="hasLayerProperty",
            Value=[ValueOperator(Name="hasLayerProperty", Op="sequal", Value=["Residual"])],
            HashOn="name"
        )]
    ))

    # --- Compound Criteria ---
    test_criterias.append(Criteria(
        Name="Dense Layers with Softmax",
        Searchs=[
            SearchOperator(
                Name="hasLayer",
                Value=[ValueOperator(Name="hasLayer", Op="sequal", Value=["Dense"])],
                HashOn="value"
            ),
            SearchOperator(
                Name="hasActivationFunction",
                Value=[ValueOperator(Name="hasActivationFunction", Op="sequal", Value=["Softmax"])],
                HashOn="value"
            )
        ]
    ))

    # --- Multi-Field Clustered Criteria ---
    test_criterias.append(Criteria(
        Name="KMeans Cluster on Dropout + Units",
        Searchs=[SearchOperator(
            Name="layer_features",
            Cluster="cluster",
            Type=TypeOperator(Name="kmeans", Arguments=["3", "binary"]),
            Value=[
                ValueOperator(Name="dropout_rate", Op="name"),
                ValueOperator(Name="layer_num_units", Op="name")
            ],
            HashOn="type"
        )]
    ))

    # --- Multi-Range Conditions ---
    test_criterias.append(Criteria(
        Name="Small Networks (Units 16-128, Layers < 5)",
        Searchs=[
            SearchOperator(
                Name="layer_num_units",
                Value=[ValueOperator(Name="layer_num_units", Op="range", Value=[16, 128])],
                HashOn="found"
            ),
            SearchOperator(
                Name="layer_depth",
                Value=[ValueOperator(Name="layer_depth", Op="less", Value=[5])],
                HashOn="found"
            )
        ]
    ))

    # --- Clustering on Task Type and Loss ---
    test_criterias.append(Criteria(
        Name="Agg Cluster on Task + Loss",
        Searchs=[SearchOperator(
            Name="meta_properties",
            Cluster="cluster",
            Type=TypeOperator(Name="agg", Arguments=["2", "binary"]),
            Value=[
                ValueOperator(Name="hasTaskType", Op="name"),
                ValueOperator(Name="hasLoss", Op="name")
            ],
            HashOn="type"
        )]
    ))

    # --- Custom Model Type Examples ---
    test_criterias.append(Criteria(
        Name="GANs Only",
        Searchs=[SearchOperator(
            Name="hasModelType",
            Value=[ValueOperator(Name="hasModelType", Op="sequal", Value=["GenerativeAdversarialNetwork"])],
            HashOn="name"
        )]
    ))

    # --- Presence of Normalization ---
    test_criterias.append(Criteria(
        Name="Has BatchNorm Layer",
        Searchs=[SearchOperator(
            Name="hasLayer",
            Value=[ValueOperator(Name="hasLayer", Op="sequal", Value=["BatchNormalization"])],
            HashOn="name"
        )]
    ))

    # --- Absence of Normalization ---
    test_criterias.append(Criteria(
        Name="Missing Normalization",
        Searchs=[SearchOperator(
            Name="hasLayer",
            Value=[ValueOperator(Name="hasLayer", Op="none", Value=["BatchNormalization"])],
            HashOn="name"
        )]
    ))

    # --- EarlyStopping with < 50 Epochs ---
    test_criterias.append(Criteria(
        Name="EarlyStopping & Few Epochs",
        Searchs=[
            SearchOperator(
                Name="hasTrainingStrategy",
                Value=[ValueOperator(Name="hasTrainingStrategy", Op="sequal", Value=["EarlyStopping"])],
                HashOn="name"
            ),
            SearchOperator(
                Name="epochs",
                Value=[ValueOperator(Name="epochs", Op="less", Value=[50])],
                HashOn="value"
            )
        ]
    ))

    # --- Models With > 5 Convolutional Layers ---
    test_criterias.append(Criteria(
        Name="Conv-heavy Models",
        Searchs=[
            SearchOperator(
                Name="num_convolutional_layers",
                Value=[ValueOperator(Name="num_convolutional_layers", Op="greater", Value=[5])],
                HashOn="value"
            )
        ]
    ))

    # --- Pooling Usage ---
    test_criterias.append(Criteria(
        Name="Has Pooling Layer",
        Searchs=[SearchOperator(
            Name="hasLayer",
            Value=[ValueOperator(Name="hasLayer", Op="sequal", Value=["MaxPooling", "AveragePooling"])],
            HashOn="name"
        )]
    ))

    # --- Batch Size Range ---
    test_criterias.append(Criteria(
        Name="Batch Size 16–64",
        Searchs=[SearchOperator(
            Name="batch_size",
            Value=[ValueOperator(Name="batch_size", Op="range", Value=[16, 64])],
            HashOn="value"
        )]
    ))

    # --- Tiny Models (Low depth & unit count) ---
    test_criterias.append(Criteria(
        Name="Tiny Models",
        Searchs=[
            SearchOperator(
                Name="layer_depth",
                Value=[ValueOperator(Name="layer_depth", Op="less", Value=[4])],
                HashOn="found"
            ),
            SearchOperator(
                Name="layer_num_units",
                Value=[ValueOperator(Name="layer_num_units", Op="less", Value=[32])],
                HashOn="found"
            )
        ]
    ))

    # --- Autoencoders ---
    test_criterias.append(Criteria(
        Name="Autoencoders",
        Searchs=[SearchOperator(
            Name="hasModelType",
            Value=[ValueOperator(Name="hasModelType", Op="sequal", Value=["Autoencoder"])],
            HashOn="name"
        )]
    ))

    # --- Transformer Networks ---
    test_criterias.append(Criteria(
        Name="Transformer Architectures",
        Searchs=[SearchOperator(
            Name="hasLayer",
            Value=[ValueOperator(Name="hasLayer", Op="sequal", Value=["TransformerBlock"])],
            HashOn="name"
        )]
    ))

    # --- Missing Training Strategy ---
    test_criterias.append(Criteria(
        Name="No Training Strategy Declared",
        Searchs=[SearchOperator(
            Name="hasTrainingStrategy",
            Value=[ValueOperator(Name="hasTrainingStrategy", Op="none")],
            HashOn="found"
        )]
    ))

    # --- Very Large Networks (Extreme Unit Count) ---
    test_criterias.append(Criteria(
        Name="Extreme Unit Count",
        Searchs=[SearchOperator(
            Name="layer_num_units",
            Value=[ValueOperator(Name="layer_num_units", Op="greater", Value=[5000])],
            HashOn="found"
        )]
    ))

    # --- Graph Clustering on Architecture Features ---
    test_criterias.append(Criteria(
        Name="Graph Cluster on Layer + Strategy",
        Searchs=[SearchOperator(
            Name="graph_cluster_features",
            Cluster="cluster",
            Type=TypeOperator(Name="graph", Arguments=["3", "binary"]),
            Value=[
                ValueOperator(Name="hasLayer", Op="name"),
                ValueOperator(Name="hasTrainingStrategy", Op="name")
            ],
            HashOn="type"
        )]
    ))

    criteria_list.extend(test_criterias)
    random.seed(42)
    random.shuffle(criteria_list)

    oc = OutputCriteria(
        criteriagroup=criteria_list,
        description=ex_description,
    ).model_dump_json()

    # model='qwq:32b'
    # model = "llama3.2:latest"
    # model = ChatOllama(
    #     model=model, temperature=0.1, top_p=1, repeat_penalty=1, num_ctx=5000
    # )
    # # keep te above just in case things dont't work

    llm = load_environment_llm(temperature=0.1, num_ctx=5000)
    if not llm:
        raise Exception("ERROR loading environment llm")
    model = llm.llm

    # a fixing parser if the original model doesnt work

    # create a parser to parse output criteria
    parser = PydanticOutputParser(pydantic_object=OutputCriteria)
    criteriaprompt = ChatPromptTemplate(
        [("system", system_prompt), ("human", "{user_input}")]
    ).partial(format_instructions=parser.get_format_instructions())
    fixparser = OutputFixingParser.from_llm(parser=parser, llm=model)

    # llm.
    # llm.llm.with_structured_output(OutputCriteria)
    chain = criteriaprompt | model

    # ontology_path = f"./data/owl/annett-o-test.owl"
    # ontology_path = f"./data/owl/fairannett-o.owl"
    # ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    # for prop in ontology.get_properties():
    properties = (
        list(ontology.data_properties())
        + list(ontology.object_properties())
        + list(ontology.classes())
    )
    properties = [prop.name for prop in properties]
    print(properties)

    output = chain.invoke(
        {
            "user_input": query,
            "oc": oc,
            "schema": json.dumps(OutputCriteria.model_json_schema()),
            "valueoperator": json.dumps(ValueOperator.model_json_schema()),
            "typeoperator": json.dumps(TypeOperator.model_json_schema()),
            "properties": json.dumps(properties).replace('"', ""),
        }
    )  # .content

    print(output)

    output = output.content
    output = re.sub(r"<think>.*?</think>\n?", "", output, flags=re.DOTALL)

    # fixing the criteria -- this may fail sometimes
    thecriteria = output = fixparser.parse(output)
    handle = open("criteria.json", "w")
    handle.write(json.dumps(thecriteria.model_dump()))
    handle.close()

    return thecriteria


if __name__ == "__main__":
    # ontology_path = f"./data/owl/fairannett-o.owl"
    ontology_path = f"./data/Annett-o/annett-o-0.1.owl"
    ontology = load_annetto_ontology(return_onto_from_path=ontology_path)

    thecriteria = llm_create_taxonomy(
        "What would you say is the taxonomy that represents all neural network?",
        ontology,
    )
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
    taxonomy_creator = TaxonomyCreator(ontology, criteria=thecriteria.criteriagroup)
    format = "json"
    topnode, facetedTaxonomy, output = taxonomy_creator.create_taxonomy(
        format=format, faceted=True
    )

    # # create a dataframe
    df = create_tabular_view_from_faceted_taxonomy(
        taxonomy_str=json.dumps(serialize(facetedTaxonomy)), format="json"
    )
    df.to_csv("./test_tax_yuh.csv")
    import dataframe_image as dfi

    dfi.export(df, "./test_tax_yuh.png")
    # taxonomy_creator = TaxonomyCreator(ontology,criteria=thecriteria.criteriagroup)
    # topnode, faceted, output = taxonomy_creator.create_taxonomy(format='json', faceted=True)

    # with open('test.json', 'w') as handle:
    #     handle.write(json.dumps(serialize(faceted)))
