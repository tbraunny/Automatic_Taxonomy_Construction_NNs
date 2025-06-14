import streamlit as st
from utils.llm_service import load_environment_llm
from neo4j_connection import get_neo4j_credentials, get_driver


def chat_page():
    try:
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
        instances = [str(instance).replace('user_owl.', '') for instance in instances]
        instances = ['_'.join(instance.split('_')[1:]) for instance in instances]
        instance_names = ", ".join(instances)
        
        if not instances:
            st.markdown("""
            <div style="background-color: #ffcc00; padding: 20px; border-radius: 10px; font-family: 'Arial', sans-serif;">
                <h3 style="color: #333333; font-size: 18px; font-weight: bold;">🛑 No instances found in the ontology!</h3>
                <p style="color: #333333; font-size: 16px;">Please add instances to your ontology to interact with this system. Once instances are added, you can ask questions related to the ontology.</p>
            </div>
        """, unsafe_allow_html=True)
            
            if st.button("Click this to keep you busy!"):
        # Embed the Rickroll video
                st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ", start_time=0, autoplay = True)
            
            return
        
        # Database & LLM Setup
        url,username,password = get_neo4j_credentials()

        graph = Neo4jGraph(url=url, username=username, password=password)
        llm = load_environment_llm().llm
        #llm = OllamaLLM(model="gemma3:27b-it-q4_K_M")

        # Question validation prompt
        QUESTION_VALIDATION_TEMPLATE = f"""
        You are an assistant that checks whether a user's question is relevant and answerable based on the following ontology instances of neural network architectures:
        {instance_names}

        Your job is to find if the question is valid or invalid based on:
        - Is related to the ontology domain (based on the instance names)
        - Is well-formed and understandable
        - Can likely be answered with data from the ontology graph
        
        only if the question is invalid, provide types of valid questions based on the question provided. In addition, list out the instances.
        
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
                WHERE a.display_name = "instance_name"
                MATCH (a)-[r*1..2]->(b) 
                RETURN a, r, b;

            multiple instances:
                MATCH (a)
                WHERE a.display_name = "instance_name"
                MATCH (a)-[r*1..5]->(b)
                RETURN "instance_name" AS type, a, r, b
                
                UNION ALL

                MATCH (a)
                WHERE a.display_name = "instance_name"
                MATCH (a)-[r*1..5]->(b)
                RETURN "instance_name" AS type, a, r, b
                

        For efficiency, please consider changing the depth of the query, example:
        MATCH (a)-[r*1..2]->(b) to MATCH (a)-[r*1..5]->(b)

        If no specific instance name is relevant to the question, return:
        MATCH (a) 
        WHERE a.display_name IN ["nonexistent_instance"] 
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

        Your task is to generate a well-structured and informative answer to the question. 
        Base your response on the provided context (do not mention the provided context in your answer). 
        Reference the relevant nodes and relationships in your explanation where appropriate.
        
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
            
        st.markdown(f"""
            <h1 style="font-family: 'Arial', sans-serif; color: #fb8c00;">Chat with AI on your Ontology</h1>
            <p style="font-family: 'Arial', sans-serif; font-size: 25px; color: inherit;">
                🧠 Ask about neural network instances from the ontology! 💬<br><br>
                📌 Be sure to include these instances in your question: {instance_names}
            </p>
        """, unsafe_allow_html=True)
            
        # Display previous messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(
                f"<div style='font-size:20px; font-family:Arial;'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        
        # Initialize llm thinking to false
        if "thinking" not in st.session_state:
            st.session_state.thinking = False
            
        def disable_callback():
            st.session_state.thinking = True
            
        user_input = None
        # Chat input
        user_input = st.chat_input("Ask something about the ontology...", disabled= st.session_state.thinking, on_submit=disable_callback)

        if user_input:
            st.session_state.thinking = True
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown( f"<div style='font-size:20px; font-family:Arial;'>{user_input}</div>",
        unsafe_allow_html=True,)

            try:
                # Step 1: Validate question
                with st.spinner("Validating your input..."):
                    validation_response = validation_chain.run({"question": user_input}).strip()

                with st.chat_message("assistant"):
                    st.markdown(f"<div style='font-size:20px; font-family:Arial;'>{validation_response}</div>",
        unsafe_allow_html=True,)

                if not validation_response.endswith("This is a valid question."):
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": validation_response}
                    )
                    st.session_state.thinking = False
                    st.rerun()
                    return

                # Step 2: Process the valid query
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = graphCypher_chain.invoke({"query": user_input})
                        raw_result = answer["result"]
                        clean_result = strip_think_block(raw_result)

                        container = st.empty()

                        def stream_and_style(text):
                            output = ""
                            for line in text.splitlines(keepends=True):
                                output += line
                                container.markdown(output)  # Stream plain text
                                time.sleep(0.05)
                            
                            # Once done, replace with styled version
                            styled_output = f"<div style='font-size:20px; font-family:Arial;'>{output}</div>"
                            container.markdown(styled_output, unsafe_allow_html=True)
                            return styled_output.strip()

                        full_response = stream_and_style(clean_result)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                st.session_state.thinking = False
                st.rerun()

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.chat_message("assistant").write(error_msg)
                st.rerun()
                
        with st.container():
            if st.session_state.chat_history != []:
                st.markdown("---")
                if st.button("🧹 Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
    except Exception as e:
        print(e)
        st.error(f"An unexpected error occurred. Please try again later. 🚨")
        
        
