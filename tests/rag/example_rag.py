from utils.rag_engine import LocalRagEngine, RemoteRagEngine



def local_query():    
    pdf_path = "data/raw/AlexNet.pdf"
    llm_model='llama3.2:1b'
    query_engine = LocalRagEngine(pdf_path=pdf_path,llm_model=llm_model).get_rag_engine()
    response = query_engine.query("What is this paper about?")
    print(response)



def remote_query():
    pdf_path = 'data/raw/VGG16.pdf'
    pdf_path = 'data/raw/AlexNet.pdf'

    ip_addr = '100.105.5.55'
    ip_addr = 'localhost'
    port = 5000
    query_engine = RemoteRagEngine(pdf_path=pdf_path, device_ip=ip_addr, port=port).get_rag_engine()
    response = query_engine.query("Who is the author of this paper")
    print(response)

local_query()