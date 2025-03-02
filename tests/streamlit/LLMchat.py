import streamlit as st

def chat_page():
    
    
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

    llm = OllamaLLM(model = "llama3.1:8b-instruct-fp16") 

    llm_transformer = LLMGraphTransformer(llm=llm)

    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import PromptTemplate
    from langchain_neo4j import GraphCypherQAChain

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
    (Do not change the structure of the graph)
    
    If there is no relevant Cypher query return nothing.

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
    
    If there is no context provided, try to answer the question pertaining to aspects in the graph. 
    
    Do not mention anything about the context in the answer.
    
    

    Context: {context}
    Question: {question}
    """

    CYPHER_QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
    )

    graphCypher_chain = GraphCypherQAChain.from_llm(
        OllamaLLM(model = "llama3.1:8b-instruct-fp16", temperature=0.0), 
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
            # Replace 'graphCypher_chain.invoke' with your actual function to generate a response
            answer = graphCypher_chain.invoke({"schema": graph.get_schema, "query": user_input})
            # Display the user's message
            messages.chat_message("user").write(user_input)
            # Display the model's response
            messages.chat_message("assistant").write(f"Answer: {answer['result']}")
        except Exception as e:
            st.write("An error occurred while processing your request. Please ask questions related to the taxonomy.")
    else:
        st.write("Please enter a prompt!")

        