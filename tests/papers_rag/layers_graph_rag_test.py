import re
import time
import networkx as nx
import ollama
import chromadb

from utils.document_json_utils import load_documents_from_json
from utils.doc_chunker import semantically_chunk_documents

def embed_and_store_chunks(documents, collection):
    """
    Embed and store chunks in the vector database.
    
    :param documents: A list of chunked Document objects.
    :param collection: The vector database collection to store embeddings.
    """
    for i, doc in enumerate(documents):
        try:
            response = ollama.embeddings(model="bge-m3:latest", prompt=doc.page_content)
            embedding = response.get("embedding")
            if embedding:
                doc_id = f"chunk_{i}"
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc.page_content],
                    metadatas=[doc.metadata]  
                )
            else:
                raise ValueError(f"Embedding failed for chunk {i}")
        except Exception as e:
            print(f"Error embedding chunk {i}: {e}")


def retrieve_relevant_chunks(prompt, collection, max_chunks=30, token_budget=2048):
    """
    Retrieve the most relevant chunks for the given prompt, adhering to a token budget.
    
    :param prompt: The query prompt.
    :param collection: The vector database collection to query from.
    :param max_chunks: Maximum number of chunks to return.
    :param token_budget: Total token budget for context (not used here explicitly, but could be implemented).
    :return: A list of relevant chunk dictionaries.
    """
    try:
        response = ollama.embeddings(model="bge-m3:latest", prompt=prompt)
        embedding = response.get("embedding")
        if embedding:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=max_chunks
            )
            # Check that results include the expected keys
            if not all(isinstance(results.get(key, []), list) for key in ['documents', 'distances', 'metadatas']):
                print("Unexpected response format from vector database.")
                return []
            
            # Flatten results from the (possibly nested) response structure.
            all_docs = []
            # The structure here assumes results['documents'] etc. are lists of lists.
            for docs, scores, metas in zip(
                    results.get('documents', []),
                    results.get('distances', []),
                    results.get('metadatas', [])
                ):
                for doc, score, meta in zip(docs, scores, metas):
                    all_docs.append({"content": doc, "score": score, "metadata": meta})
            
            # Sort by score (assume higher score means more relevant)
            sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
            
            # (Optional) Here you might want to add logic to ensure token budget is not exceeded.
            context = sorted_docs[:max_chunks]
            return context
        else:
            print("Error: Failed to generate embedding for prompt.")
            return []
    except Exception as e:
        print(f"Error querying relevant documents: {e}")
        return []


def build_graph_from_documents(documents):
    """
    Build a directed graph representing the neural network from the code.
    This example uses a simple regex to detect layer definitions (e.g. "nn.Conv2d(" or "nn.Linear(").
    
    :param documents: List of Document objects containing code.
    :return: A networkx DiGraph where each node is a layer name.
    """
    G = nx.DiGraph()
    layer_pattern = r"(nn\.[A-Za-z0-9_]+)\("  # Adjust regex for your framework/language if needed.
    
    for doc in documents:
        code = doc.page_content
        layers = re.findall(layer_pattern, code)
        if layers:
            previous_layer = None
            for layer in layers:
                # Add node if it does not exist
                if not G.has_node(layer):
                    G.add_node(layer)
                if previous_layer:
                    # Add an edge from the previous layer to the current one.
                    G.add_edge(previous_layer, layer)
                previous_layer = layer
        else:
            print("No layer definitions found in one of the documents.")
    return G


def get_ordered_layers_from_graph(G):
    """
    Return the list of layers in order by performing a topological sort.
    This assumes that the network definition is a directed acyclic graph.
    
    :param G: networkx DiGraph representing the network.
    :return: List of layer names in sequential order.
    """
    try:
        ordered_layers = list(nx.topological_sort(G))
        return ordered_layers
    except Exception as e:
        print(f"Error in ordering layers from graph: {e}")
        return []


# ------------------------
# COMBINED RESPONSE GENERATION
# ------------------------
def generate_optimized_response(prompt, vector_context, graph_context):
    """
    Generate a concise and contextually relevant response using an LLM,
    combining both vector-retrieved semantic chunks and graph-derived layer ordering.
    
    :param prompt: The original user prompt.
    :param vector_context: List of context chunks retrieved from the vector database.
    :param graph_context: Ordered list of layer names derived from the graph.
    :return: A generated response string.
    """
    try:
        # Prepare the vector context string.
        vector_context_str = "\n\n".join([
            f"### Chunk Metadata: {chunk['metadata']}\n{chunk['content']}"
            for chunk in vector_context
        ])
        
        # Prepare the graph context string.
        graph_context_str = " -> ".join(graph_context) if graph_context else "No graph context available."
        
        # Build the final instructions including both context sources.
        instructions = (
            f"Given the following information extracted from the codebase:\n\n"
            f"---\n"
            f"**Graph-based layer order:**\n{graph_context_str}\n\n"
            f"---\n"
            f"**Additional semantic context from code chunks:**\n{vector_context_str}\n\n"
            f"Answer the following prompt in a concise and structured manner:\n"
            f"'{prompt}'\n\n"
            f"List each layer sequentially by name. Do not generalize internal layers; include all types such as modification and activation layers."
        )
        
        print("### Instructions Provided to LLM ###")
        print(instructions)
        print("#######################################")
        
        response = ollama.generate(
            model="deepseek-r1:32b",
            prompt=instructions,
            options={"num_ctx": 3000},
        )
        return response.get('response', "No response generated.")
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error in response generation."


def main():
    client = chromadb.Client()
    collection = client.create_collection(name="file_docs")
    
    json_file_path = "/home/richw/richie/Automatic_Taxonomy_Construction_NNs/data/alexnet/alexnet_code0.json"
    
    # Load documents from JSON.
    docs = load_documents_from_json(json_file_path)
    
    chunked_docs = semantically_chunk_documents(docs)
    
    embed_and_store_chunks(chunked_docs, collection)
    
    # Build a graph representation from the full documents
    graph = build_graph_from_documents(docs)
    ordered_layers = get_ordered_layers_from_graph(graph)
    
    # Prepare the user prompt.
    prompt = (
        "Give each layer of this neural network sequentially. Do not generalize internal layers and include all types of layers such as modification and activation layers."
    )
    
    # Retrieve vector-based relevant context.
    vector_context = retrieve_relevant_chunks(prompt, collection, max_chunks=30, token_budget=2048)
    
    # Generate the final answer using both vector and graph contexts.
    if vector_context or ordered_layers:
        start_time = time.time()
        final_response = generate_optimized_response(prompt, vector_context, ordered_layers)
        print("\nFinal Answer:\n", final_response, "\nIn Time:", time.time() - start_time)
    else:
        print("No relevant documents or graph context found.")

if __name__ == "__main__":
    main()
