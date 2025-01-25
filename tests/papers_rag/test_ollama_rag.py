import os
import ollama
import chromadb
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize ChromaDB client and create a collection
client = chromadb.Client()
collection = client.create_collection(name="file_docs")

def combine_header_and_paragraphs(data, max_chunk_size=1000, overlap=200):
    """
    Combine headers and their associated paragraphs into meaningful chunks,
    ensuring the header applies to all paragraphs in the section.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=overlap
    )

    documents = []
    for section in data:
        header = section["header"]
        content = "\n".join(section["content"]).strip()  # Combine all paragraphs
        # Split content into chunks
        chunks = text_splitter.split_text(content)
        for chunk in chunks:
            # Store each chunk with its associated header as metadata
            documents.append(Document(page_content=chunk, metadata={"header": header}))
    return documents

def read_json_files(json_file_path):
    """
    Read a JSON file containing parsed sections with headers and content.
    """
    import json
    with open(json_file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def embed_and_store_chunks(documents, collection):
    """
    Embed and store chunks in the vector database.
    """
    for i, doc in enumerate(documents):
        try:
            response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=doc.page_content)
            embedding = response.get("embedding")
            if embedding:
                doc_id = f"chunk_{i}"
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata]  # Include metadata
                )
            else:
                print(f"Failed to embed chunk {i}")
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")

def retrieve_relevant_chunks(prompt, collection, max_chunks=10, token_budget=1024):
    """
    Retrieve the most relevant chunks for the given prompt, adhering to a token budget.
    """
    try:
        response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=prompt)
        embedding = response.get("embedding")
        if embedding:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=max_chunks
            )
            all_docs = [
                {"content": doc, "score": score, "metadata": meta}
                for docs, scores, metas in zip(
                    results.get('documents', []),
                    results.get('distances', []),
                    results.get('metadatas', [])
                )
                for doc, score, meta in zip(docs, scores, metas)
            ]
            
            # Sort chunks by relevance (lower distance = higher relevance)
            sorted_docs = sorted(all_docs, key=lambda x: x["score"])
            
            # Concatenate chunks within the token budget
            context = []
            total_tokens = 0
            for doc in sorted_docs:
                doc_tokens = len(doc["content"].split())  # Rough token estimation
                if total_tokens + doc_tokens <= token_budget:
                    context.append(doc)
                    total_tokens += doc_tokens
                else:
                    break
            return context
        else:
            print("Error: Failed to generate embedding for prompt.")
            return []
    except Exception as e:
        print(f"Error querying relevant documents: {e}")
        return []

def format_response(relevant_chunks):
    """
    Format retrieved chunks into a coherent response.
    """
    grouped = {}
    for chunk in relevant_chunks:
        header = chunk["metadata"].get("header", "No Header")
        if header not in grouped:
            grouped[header] = []
        grouped[header].append(chunk["content"])

    response = []
    for header, contents in grouped.items():
        response.append(f"### {header}\n\n" + "\n\n".join(contents))
    return "\n\n---\n\n".join(response)

def generate_optimized_response(prompt, context):
    """
    Generate a concise and contextually relevant response using an LLM.
    """
    try:
        # Combine the retrieved context
        full_context = "\n\n".join([chunk["content"] for chunk in context])
        
        # Print the context being passed to the LLM
        print("### Context Provided to LLM ###")
        print(full_context)
        print("################################")
        
        # Generate the response
        response = ollama.generate(
            # model="llama3.1:8b-instruct-fp16",
            model="deepseek-r1:14b",
            prompt=(
                f"Using this context:\n{full_context}\n\n"
                f"Answer the following question concisely and accurately:\n{prompt}"
            )
        )
        return response.get('response', "No response generated.")
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error in response generation."

# Example Usage

# Path to JSON file containing parsed headers and content
json_file_path = "data/alexnet/parsed_alexnet.json"

# Step 1: Read and preprocess the JSON data
parsed_data = read_json_files(json_file_path)
documents = combine_header_and_paragraphs(parsed_data)

# Step 2: Embed and store documents
embed_and_store_chunks(documents, collection)

# Step 3: Retrieve relevant chunks for a prompt
prompt = "Describe the methodology used in the research to reduce overfitting, including any data augmentation techniques or regularization strategies."
max_chunks = 10
token_budget = 1024
relevant_chunks = retrieve_relevant_chunks(prompt, collection, max_chunks=max_chunks, token_budget=token_budget)

# Step 4: Format and generate the final response
if relevant_chunks:
    formatted_response = format_response(relevant_chunks)
    final_response = generate_optimized_response(prompt, relevant_chunks)
    print("Formatted Response:\n", formatted_response)
    print("\nFinal Answer:\n", final_response)
else:
    print("No relevant documents found.")
