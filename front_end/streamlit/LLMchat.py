import streamlit as st
from instances import list_of_class_instances

def chat_page():
    import re
    import time
    from langchain_neo4j import Neo4jGraph
    from neo4j import GraphDatabase
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain_neo4j import GraphCypherQAChain

    # Helper to remove internal thoughts
    def strip_think_block(text):
        """Remove <think>...</think> blocks from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Get ontology instance names
    instances = list_of_class_instances()
    instance_names = ", ".join([instance.name for instance in instances])

    # Database & LLM Setup
    url = "bolt://localhost:7687"
    username = "neo4j"
    password = "neo4j"
    graph = Neo4jGraph(url=url, username=username, password=password)
    driver = GraphDatabase.driver(url, auth=(username, password))
    session = driver.session()
    llm = OllamaLLM(model="deepseek-r1:32b")

    # Cypher generation prompt
    CYPHER_GENERATION_TEMPLATE = f"""
    You are an AI system that generates Cypher queries for retrieving data from a graph database. 
    The graph schema consists of nodes and relationships related to neural network ontologies. 
    These are the exact instance names: {instance_names}

    You will receive a question related to this domain. Based on the question, generate a Cypher query that answers it.

    Cypher Query will look like:
    MATCH (a)
    WHERE a.uri IN ["instance name"]
    MATCH (a)-[r*1..7]->(b)
    RETURN a, r, b

    insert the correct instance name relevant to the question.

    Question: {{question}}
    """

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
    )

    # QA prompt
    CYPHER_QA_TEMPLATE = f"""
    You are an assistant that helps to provide human-readable answers to questions based on classes and relationships in the graph.
    The graph is related to ontology instances on: {instance_names}

    You will receive the context (results of a Cypher query) and a question. Based on that, return an understandable answer.

    Do not mention anything about the context in the answer.

    Context: {{context}}
    Question: {{question}}
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
        allow_dangerous_requests=True
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with AI on Ontology")
    st.write("Ask about neural network classes or instances from the ontology.")

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about the ontology...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = graphCypher_chain.invoke({"query": user_input})
                    raw_result = answer["result"]
                    clean_result = strip_think_block(raw_result)

                    streamed_output = []

                    # Stream line-by-line for proper markdown rendering
                    def stream_text(text):
                        for line in text.splitlines(keepends=True):
                            streamed_output.append(line)
                            yield line
                            time.sleep(0.05)

                    st.write_stream(stream_text(clean_result))

                    # Save final output to chat history
                    full_response = "".join(streamed_output).strip()
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": full_response}
                    )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.chat_history.append(
                {"role": "assistant", "content": error_msg}
            )
            st.chat_message("assistant").write(error_msg)
