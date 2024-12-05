from utils.rag_engine import LocalRagEngine, RemoteRagEngine
from utils.llm_model import OllamaLLMModel



def local_ollama():


    from utils.pdf_loader import load_pdf
    file_path = "data/hand_processed/AlexNet.pdf"  # Replace with your actual file path
    documents = load_pdf(file_path)
    # Combine all the page contents into a single string
    paper_content = "\n".join([doc.page_content for doc in documents])
    model_name = 'llama3.2:3b-instruct-fp16'
    llm = OllamaLLMModel(temperature=0.5,top_k=7,model_name=model_name)

    instructions = f"""Consider the following document delimited by triple backticks as context to the question ```{paper_content}``` Question: """
    instructions += """{query}"""
    query = """How does this nueral network architecture account for overfitting"""

    num_tokens_instruct = llm.count_tokens(instructions)
    num_tokens_query = llm.count_tokens(query)
    print(f"Number of tokens = {num_tokens_instruct + num_tokens_query}")

    # response = llm.query_ollama(query,instructions)
    # print(response)




def local_rag_query():    
    pdf_path = "data/raw/AlexNet.pdf"
    model_name = 'llama3.2:3b-instruct-fp16'
    model = LocalRagEngine(pdf_path=pdf_path)#,llm_model=model_name)

    prompt = """How does this nueral network architecture account for overfitting"""

    rag_chunks = model.get_relevant_chunks(prompt)
    print(rag_chunks)

    print("\n\n\n")

    rag_engine = model.get_rag_engine()
    response = rag_engine.query(prompt)
    print(response)


def remote_rag_query():
    pdf_path = 'data/raw/VGG16.pdf'
    pdf_path = 'data/raw/AlexNet.pdf'

    ip_addr = '100.105.5.55'
    ip_addr = 'localhost'
    port = 5000
    query_engine = RemoteRagEngine(pdf_path=pdf_path, device_ip=ip_addr, port=port).get_rag_engine()
    response = query_engine.query("Who is the author of this paper")
    print(response)
