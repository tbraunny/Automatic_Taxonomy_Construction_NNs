import os
import time

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_community.chat_models import ChatOllama
#from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

from langchain.prompts import PromptTemplate # template prompts
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain
#from langchain_experimental.graph_transformers import LLMGraphTransformer # model Import
from langchain_ollama.llms import OllamaLLM

def neo4j_connection():
    url = "bolt://localhost:7687" # neo4j connector
    username = "neo4j"
    password= "neo4j"

    graph = Neo4jGraph(url=url, username=username, password=password) 
    driver = GraphDatabase.driver(url, auth=(username, password))
    session = driver.session()
    return graph, session,driver

def queryNeo4j(driver, query):
    """Runs a single Cypher query."""
    with driver.session() as session:
        try:
            return session.run(query)
            
        except Exception as e:
            print(f"Error executing query: {e}")


CYPHER_GENERATION_TEMPLATE = """
You are an AI system that generates Cypher queries to retrieve data from a graph database. 
The graph schema consists of nodes and relationships related to the following ontologies:
- Adversarial Autoencoders (AAEs)
- Generative Adversarial Networks (GANs)
- Simple Classification
You will receive a question related to the domain. Based on that, generate a Cypher query that answers the question.

Here is the schema:
{schema}

For example if you are to retrieve the nodes to a specific naming convention:

MATCH (n)
WHERE n.uri IS NOT NULL AND n.uri CONTAINS 'GAN'
RETURN n

Based on the above, answer the following question by creating a Cypher query to retrieve the relevant data from the graph.

Question: {question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

CYPHER_QA_TEMPLATE = """
You are an assistant that helps to provide human-readable answers to questions based on classes and relationships in the graph.
The graph is related to three ontology instances on:
- Adversarial Autoencoders (AAEs)
- Generative Adversarial Networks (GANs)
- Simple Classification

You will receive the context (results of a Cypher query) and a question. Based on that, return an understandable answer.

Context: {context}
Question: {question}
"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)



def get_schema_from_neo4j(graph):
    schema = graph.get_schema
    return schema

def get_answer_for_question(graph, question, driver):
    schema = get_schema_from_neo4j(graph)
    
    graphCypher_chain = GraphCypherQAChain.from_llm(
        OllamaLLM(model = "deepseek-r1:32b-qwen-distill-q4_K_M", temperature=0.0), 
        graph=graph, 
        cypher_prompt=CYPHER_GENERATION_PROMPT, 
        qa_prompt=CYPHER_QA_PROMPT,
        verbose=True,
        allow_dangerous_requests = True
    )

    cypher_query = graphCypher_chain.invoke({
        "schema" : schema,
        "query" : question
    })
    
    cypher_query_string = cypher_query.get("query")
    
    if not cypher_query_string:
        raise ValueError("Cypher query generation failed or returned an empty query.")
    
    context = queryNeo4j(driver, cypher_query)
    
    human_readable_answer = graphCypher_chain.invoke({
        "context" : context,
        "query" : question
    })
    
    return human_readable_answer

if __name__ == "__main__": 
    graph, session, driver = neo4j_connection()
    question = "what networks are there?" # change this to get a question from graphRAG

    answer = get_answer_for_question(graph, question,driver)
    print(answer)
