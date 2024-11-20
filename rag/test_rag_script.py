from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.langchain import LangchainEmbedding

import json

def load_pdf(file_path: str) -> list:
    """
    Loads the PDF into document objects.
    :return: List of documents loaded from the PDF.
    :rtype: list
    """
    if file_path is not None:
        documents = []
        print("Loading PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF.")
        return documents
    else:
        print('\nPlease provide file path.')
        return None


def chunk_document(documents, chunk_size=1000,chunk_overlap=200) -> list:
    """
    Splits a list of documents into smaller chunks.
    :param documents: List of documents to split.
    :type documents: list
    :param chunk_size: Maximum size of each chunk.
    :type chunk_size: int
    :param chunk_overlap: Overlap size between chunks.
    :type chunk_overlap: int
    :return: List of document chunks.
    :rtype: list
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs


MODEL_NAME = 'all-MiniLM-L6-v2'

class EmbeddingModel:
    """
    A utility class for initializing and retrieving an embedding model.
    """
    def __init__(self, model_name=MODEL_NAME):
        """
        Constructor for embedding model.
        :param model_name: Name of the HuggingFace embedding model.
        :type model_name: string
        """
        print("Initializing the embedding model...")
        self.embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name=model_name)
        )
        print("Embedding model initialized.")

    def get_model(self):
        """
        Retrieves the embedding model.
        :return: The embedding model object.
        :rtype: LangchainEmbedding
        """
        return self.embed_model


MODEL_NAME = 'llama3.2:1b'

class LLMModel:
    """
    A utility class for initializing and retrieving a large language model (LLM).
    """
    def __init__(self, model_name: str=MODEL_NAME, top_p:float=0.9, temperature:float=0, top_k:int=3, max_tokens:int=10):
        """
        Constructor for LLM model.
        :param model_name: Name of the LLM model.
        :type model_name: string
        :param top_p: Top-p (nucleus sampling) value for generation.
        :type top_p: float
        :param temperature: Sampling temperature for generation.
        :type temperature: float
        :param top_k: Top-k sampling parameter for generation.
        :type top_k: int
        """
        print("Initializing the LLM...")
        self.llm_predictor = LangChainLLM(
            llm=OllamaLLM(
                model=model_name,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens
                )
            )

    def get_llm(self):
        """
        Retrieves the LLM predictor object.
        :return: The LLM predictor object.
        :rtype: LangChainLLM
        """
        return self.llm_predictor


class DocumentIndexer:
    """
    A utility class for creating and managing a document index.
    """
    def __init__(self, embed_model, llm_model,documents):
        """
        Constructor for DocumentIndexer.
        :param embed_model: Embedding model object.
        :type embed_model: LangchainEmbedding
        :param llm_predictor: LLM predictor object.
        :type llm_predictor: LangChainLLM
        """
        self.embed_model = embed_model
        self.llm_model = llm_model
        # Set the global settings for LLM and embedding model
        Settings.llm = self.llm_model
        Settings.embed_model = self.embed_model
        self.vector_index = None
        self.create_index(documents)

    def create_index(self, documents):
        """
        Creates a VectorStoreIndex from a list of documents.
        :param documents: List of documents to index.
        :type documents: list
        :return: VectorStoreIndex object.
        :rtype: VectorStoreIndex
        """
        print("Creating LlamaIndex documents...")
        index_documents = [Document(text=doc.page_content) for doc in documents]
        print(f"Created {len(index_documents)} LlamaIndex documents.")

        print("Building the VectorStoreIndex...")
        self.vector_index = VectorStoreIndex.from_documents(
            index_documents, 
            embed_model=self.embed_model, 
            llm_predictor=self.llm_model
        )
        print("VectorStoreIndex built.")
        # return self.vector_index
    
    def get_rag_query_engine(self):
        return self.vector_index.as_query_engine()

def load_ontology_questions(json_path: str) -> dict:
    """
    Loads ontology questions from a JSON file.
    :param json_path: Path to the JSON file containing ontology questions.
    :type json_path: str
    :return: Dictionary of ontology questions.
    :rtype: dict
    """
    with open(json_path, 'r') as file:
        questions = json.load(file)
    return questions

def get_question(entity: str, entity_type: str, questions: dict) -> str:
    """
    Retrieves a question for a given entity and type (Classes, ObjectProperties, or DataProperties).
    :param entity: The entity name (e.g., 'Dataset', 'batch_size').
    :type entity: str
    :param entity_type: The type of entity (e.g., 'Classes', 'ObjectProperties', 'DataProperties').
    :type entity_type: str
    :param questions: Dictionary of ontology questions.
    :type questions: dict
    :return: The associated question or a default message if not found.
    :rtype: str
    """
    return questions.get(entity_type, {}).get(entity, f"No question found for '{entity}' in '{entity_type}'.")

def main():
    # Load and process the PDF
    pdf_path = "data/papers/AlexNet.pdf"
    # pdf_path = "data/papers/ResNet.pdf"

    print("Initializing the RAG pipeline...")

    documents = load_pdf(pdf_path)

    chunked_docs = chunk_document(documents)

    # Step 3: Initialize Embedding Model and LLM
    embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
    llm_model = LLMModel(model_name="llama3.1:8b").get_llm()

    # # Set the global settings for LLM and embedding model
    # Settings.llm = llm_model
    # Settings.embed_model = embed_model

    # Step 4: Create the Document Index
    indexer = DocumentIndexer(embed_model,llm_model,chunked_docs)
    query_engine = indexer.get_rag_query_engine()

    # Load questions from the ontology JSON file
    json_path = "rag/ontology_prompts.json"
    questions_dict = load_ontology_questions(json_path)

    # Retrieve a question for a class
    class_name = "num_of_iterations"
    object_type = "DataProperties"

    # class_name = "hasLayer"
    # object_type = "ObjectProperties"

    question = get_question(class_name, object_type, questions_dict)
    print(f"Question for class '{class_name}': {question}")

    initial_prompt = """
                    Work out your chain of thought.
                    If you do not know the answer to a question, respond with "N/A." and nothing else.
                    Question:
                    """

    query = "".join([initial_prompt,question])

    response = query_engine.query(query)
    second_prompt = f"""
                Given the question "{question}" and it's response "{response}", rephrase the response to follow these formatting rules:
                Single-item answers: If the question requires only one answer, respond with just that single answer.
                Single-value answers: If the question requires only one value, respond with just that single value.
                Listed answers: If the question requires multiple answers without order of priority, provide them as a list, with each answer as a single, complete item.
                Numbered list answers: If the question requires multiple answers in a specific sequence or hierarchy, provide them as a numbered list, with each number followed by a single, complete item.
                Use atomic answers for clarity, meaning each response should contain only one idea or concept per point. Ensure the format aligns with the nature of the question being asked.
                If multiple questions is asked, respond with a list in the order the questions were asked and nothing else. Do not label your answers.
                If the response says "N/A" or suggests that it does not know, reply with "N/A" and nothing else.

                Single-item answer example:
                Question: What is the capital of France?
                Answer: Paris

                Single-vlaue answer example:
                Question: How many states are in the United States?
                Answer: 50

                Listed answers example:
                Question: What are the primary colors in the American Flag?
                Answer: Red, White, Blue

                Numbered list example:
                Question: What are the steps in painting a wall?
                Answer: collect tools, 1 coat wall, 2 coat wall, 3 coat wall, wait to dry
                """
    
    final_response = query_engine.query(second_prompt) 
    print("\nAnswer:")
    print("-" * 50)
    print(response)
    print("-" * 50)
    print("\nAnswer:")
    print("-" * 50)
    print(final_response)
    print("-" * 50)

if __name__ == "__main__":
    main()
