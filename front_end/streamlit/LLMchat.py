import streamlit as st

def chat_page():
    import re
    import time
    from langchain_neo4j import Neo4jGraph
    from neo4j import GraphDatabase
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_ollama.llms import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_neo4j import GraphCypherQAChain
    from front_end.instances import list_of_class_instances

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
    llm = OllamaLLM(model="gemma3:27b-it-q4_K_M")

    # Question validation prompt
    QUESTION_VALIDATION_TEMPLATE = f"""
    You are an assistant that checks whether a user's question is relevant and answerable based on the following ontology instances of neural network architectures:
    {instance_names}

    Your job is to find if the question is valid or invalid based on:
    - Is related to the ontology domain (based on the instance names)
    - Is well-formed and understandable
    - Can likely be answered with data from the ontology graph
    
    only if the question is invalid, provide types of valid questions based on the instance names.
    
    end your answer with one of these two lines:
    ✅ This is a valid question.
    ❌ This is not a valid question.

    Question: {{question}}
    """

    VALIDATION_PROMPT = PromptTemplate(
        input_variables=["question"], template=QUESTION_VALIDATION_TEMPLATE
    )
    validation_chain = LLMChain(llm=llm, prompt=VALIDATION_PROMPT)

    # Cypher generation prompt
    CYPHER_GENERATION_TEMPLATE = f"""
    You are an AI system that generates Cypher queries for retrieving data from a graph database. 

    These are the exact instance names: {instance_names}

    You will receive a question related to this domain. Based on the question insert the correct instance name relevant to the question.

    Example Cypher Queries:
        single instances:
            MATCH (a) 
            WHERE a.uri CONTAINS "instance_name"
            MATCH (a)-[r*1..2]->(b) 
            RETURN a, r, b;

        multiple instances:
            MATCH (a) 
            WHERE a.uri CONTAINS "instance_name1" or a.uri CONTAINS "instance_name2"
            MATCH (a)-[r*1..2]->(b) 
            RETURN a, r, b;

    For efficiency, please consider changing the depth of the query, example:
    MATCH (a)-[r*1..2]->(b) to MATCH (a)-[r*1..5]->(b)

    If no specific instance name is relevant to the question, return:
    MATCH (a) 
    WHERE a.uri IN ["nonexistent_instance"] 
    RETURN a

    Question: {{question}}
    """

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
    )

    # QA prompt
    CYPHER_QA_TEMPLATE = f"""
    You are a helpful assistant providing clear, natural-language answers based on a graph of ontology instances on neural network architectures specifically: {instance_names}.

    You’ll be given the results of a Cypher query (as context) and a user question.

    Your task is to generate a well-structured and informative answer to the question. Base your response on the provided context. Reference the relevant nodes and relationships in your explanation where appropriate, but do not mention that the data comes from a graph or Cypher.

    Focus on clarity, completeness, and relevance to the question.

    Context:
    {{context}}

    Question:
    {{question}}
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

    st.title("Chat with AI on your Ontology")
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
            # Step 1: Validate question
            with st.spinner("Validating your input..."):
                validation_response = validation_chain.run({"question": user_input}).strip()

            with st.chat_message("assistant"):
                st.markdown(validation_response)

            if not validation_response.endswith("This is a valid question."):
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": validation_response}
                )
                return

            # Step 2: Process the valid query
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = graphCypher_chain.invoke({"query": user_input})
                    raw_result = answer["result"]
                    clean_result = strip_think_block(raw_result)

                    streamed_output = []

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
