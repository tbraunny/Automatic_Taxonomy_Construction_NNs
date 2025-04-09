import streamlit as st
from instances import list_of_class_instances

def chat_page():
    #annetto class instances
    instances = list_of_class_instances()
    instance_names = ", ".join([instance.name for instance in instances])
    
    from langchain_neo4j import Neo4jGraph
    from neo4j import GraphDatabase
    url = "bolt://localhost:7687"
    username = "neo4j"
    password= "neo4j"

    graph = Neo4jGraph(url=url, username=username, password=password)
    driver = GraphDatabase.driver(url, auth=(username, password))
    session = driver.session()
    
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_ollama.llms import OllamaLLM

    llm = OllamaLLM(model = "deepseek-r1:32b") 

    from langchain.prompts import PromptTemplate
    from langchain_neo4j import GraphCypherQAChain
    
    CYPHER_GENERATION_TEMPLATE ="""
    You are an AI system that generates Cypher queries for retrieving data from a graph database. 
    The graph schema consists of nodes and relationships related to neural network ontologies. 
    These are the exact instance names:
    """ + f"{instance_names}" + """
    
    You will receive a question related to this domain. Based on the question, generate a Cypher query that answers it. 

    Cypher Query will look like:
    MATCH (a)
    WHERE a.uri IN ["instance name"]
    MATCH (a)-[r*1..7]->(b)
    RETURN a, r, b

    insert the correct instance name relevent to the question.

    example of inserting instance name:
    MATCH (a)
    WHERE a.uri IN ["GAN"]
    MATCH (a)-[r*1..7]->(b)
    RETURN a, r, b
    
    ### Important Notes:
    - If the question doesn't correspond to any relevant data in the graph, return nothing (i.e., an empty result).

    Question: {question}
    """
   
    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
    )
    
    CYPHER_QA_TEMPLATE = """
    You are an assistant that helps to provide human-readable answers to questions based on classes and relationships in the graph.
    The graph is related to ontology instances on: """ + f"{instance_names}" + """
    
    You will receive the context (results of a Cypher query) and a question. Based on that, return an understandable answer.
    
    If there is no context provided, try to answer the question pertaining to aspects in the graph. 
    
    Do not mention anything about the context in the answer.
    
    

    Context: {context}
    Question: {question}
    """

    CYPHER_QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
    )
    
    graphCypher_chain = GraphCypherQAChain.from_llm(
        llm, 
        graph=graph, 
        cypher_prompt=CYPHER_GENERATION_PROMPT, 
        qa_prompt=CYPHER_QA_PROMPT,
        verbose=True,
        allow_dangerous_requests = True
    )

    st.title("Chat with AI")
    st.write("Enter a prompt, and the model will generate a response.")
    messages = st.container()

    # Create a chat input field
    user_input = st.chat_input("Enter your prompt:")
    
    if user_input:
        try:
            answer = graphCypher_chain.invoke({
                "query": user_input
            })
            # Display the user's message
            messages.chat_message("user").write(user_input)
            # Display the model's response
            messages.chat_message("assistant").write(f"Answer: {answer['result']}")
        except Exception as e:
            st.write(f"An error occurred: {str(e)}. Please ask questions related to the taxonomy.")

        

        