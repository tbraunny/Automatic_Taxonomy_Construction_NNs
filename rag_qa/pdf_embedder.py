import asyncio
from langchain_community.document_loaders import PyPDFLoader
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("Out mf imports")

#This implementation stores in memory, skips the storing in database part

#Loads pdf to embed; placeholder until we can preprocess 
async def load_pdf():
    print("Loading pdf...")
    loader = PyPDFLoader("test_documents/AlexNet.pdf")
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)

    # print(f"{pages[0].metadata}\n")
    # print(pages[0].page_content)
    print("Finsihed loading pdf")

    return pages
# Citation: https://python.langchain.com/docs/how_to/document_loader_pdf/

#Load and encode PDF pages, then index them
async def main():
    #Load PDF pages
    pages = await load_pdf()
    print("PDF loaded")

    #Initialize the HuggingFace embedding model (This is small enough for my macbook)
    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    print("Embedding model loaded")

    #Create LlamaIndex documents without embeddings
    documents = [Document(text=page.page_content) for page in pages]

    #Initialize the vector store index with the embedding model
    vector_store = VectorStoreIndex.from_documents(documents, embed_model=embed_model) #Must specify model otherwise openai will be used
    print(vector_store)
    print("Documents indexed")

    #This is where we could initialize LLM
    #Perform a query
    #Print the response

#Run async
asyncio.run(main())

#https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/ #local embeddings w/ huggingface
#Citation: https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/
#Above link has example of how to store embendding in database (pinecone)
#https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/ #Llama index embedding guide

#llamaindex guide to local llama quries https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/

