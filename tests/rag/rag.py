from utils.rag_engine import LocalRagEngine, RemoteRagEngine



def ollama_query(query, instructions):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM

    prompt = ChatPromptTemplate.from_template(instructions)

    model = OllamaLLM(model='llama3.1:8b', num_ctx=128000, max_tokens=128000)

    chain = prompt | model

    return chain.invoke({"query": query})


def local_rag_query():    
    pdf_path = "data/raw/AlexNet.pdf"
    llm_model='llama3.2:3b-instruct-fp16'
    model = LocalRagEngine(pdf_path=pdf_path,llm_model=llm_model)

    prompt = """What is this paper about? Tell me about how it reduces overfitting"""

    # rag_chunks = model.get_relevant_chunks(prompt)
    # print(rag_chunks)

    print("\n\n\n")

    rag_engine = model.get_rag_engine()
    response = rag_engine.query(prompt)
    print(response)

local_rag_query()


def remote_rag_query():
    pdf_path = 'data/raw/VGG16.pdf'
    pdf_path = 'data/raw/AlexNet.pdf'

    ip_addr = '100.105.5.55'
    ip_addr = 'localhost'
    port = 5000
    query_engine = RemoteRagEngine(pdf_path=pdf_path, device_ip=ip_addr, port=port).get_rag_engine()
    response = query_engine.query("Who is the author of this paper")
    print(response)
