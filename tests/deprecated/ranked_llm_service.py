#!/usr/bin/env python3
"""
llm_service.py

An improved Retrieval-Augmented Generation (RAG) pipeline that incorporates several state-of-the-art methods:
  - Two-stage retrieval: Dense vector search combined with BM25-based sparse retrieval.
  - Logical routing between a vector (dense) database and a graph database.
  - LLM-based re-ranking.
  - Fusion-in-Decoder style context assembly with precise token budgeting.
  - Iterative retrieval expansion via LLM feedback.
  
Dependencies include:
  - `ollama` for LLM calls,
  - `chromadb` for the vector database,
  - `rank_bm25` for BM25 retrieval,
  - `networkx` for simulating a graph database,
  - LangChain’s Document abstraction and RecursiveCharacterTextSplitter,
  - A custom JSON document loader (`load_documents_from_json`).
"""

import json
import logging
import time
import re

import ollama
import chromadb
from rank_bm25 import BM25Okapi
import networkx as nx

from utils.doc_chunker import semantically_chunk_documents
from utils.document_json_utils import load_documents_from_json

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm_json_instructions(query: str, context: str) -> str:
    """
    Build a Fusion-in-Decoder style prompt that instructs the LLM to consider each evidence block.
    """
    instructions = (
        "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
        "Below are several context sections, each starting with a header. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
        "### Evidence Blocks:\n"
        f"{context}\n\n"
        f"Query: {query}\n"
        "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
        "with your answer listed as the value. For example, the last line of your output should be:\n"
        """ {"answer": ["name_1","name_2","name_3"]}"""
    )
    return instructions


class LLMQueryEngine:
    """
    A LLM engine that implements a multi-stage retrieval-augmented generation pipeline:
      1. Loads and chunks documents (preserving metadata such as headers).
      2. Embeds and stores chunks in a vector database (dense retrieval).
      3. Builds a BM25 index for sparse retrieval.
      4. Builds a graph index to support graph-based retrieval.
      5. Retrieves candidate chunks via logical routing between the dense/BM25 (hybrid) retrieval and graph retrieval.
      6. Re-ranks candidates using an LLM to assess relevance.
      7. Iteratively expands context if the initial retrieval does not fill the token budget.
      8. Generates a final response using a Fusion-in-Decoder style prompt.
    """

    def __init__(self,
                 json_file_path: str,
                 collection_name: str = "nn_docs",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "bge-m3",
                 generation_model: str = "deepseek-r1:32b",
                 relevance_generation_model: str = "deepseek-r1:8b"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.relevance_generation_model = relevance_generation_model

        # Initialize the vector database.
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        logger.info("Initialized vector DB collection: %s", collection_name)

        # Load and process documents.
        docs = load_documents_from_json(json_file_path)
        logger.info("Loaded %d documents from %s", len(docs), json_file_path)

        self.chunked_docs = self._chunk_documents(docs)
        logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

        self._embed_and_store_chunks(self.chunked_docs)
        logger.info("Finished embedding and storing document chunks.")

        # Build BM25 index for sparse retrieval.
        self.bm25_corpus = [doc.page_content for doc in self.chunked_docs]
        self.bm25_tokens = [doc.page_content.lower().split() for doc in self.chunked_docs]
        self.bm25_index = BM25Okapi(self.bm25_tokens)
        logger.info("Built BM25 index for %d document chunks.", len(self.chunked_docs))

        # Build a simple graph database index using networkx.
        self.graph = nx.Graph()
        for i, doc in enumerate(self.chunked_docs):
            self.graph.add_node(i, content=doc.page_content, metadata=doc.metadata)
        # Add edges between nodes that share the same section header.
        sections = {}
        for i, doc in enumerate(self.chunked_docs):
            header = doc.metadata.get("section_header")
            if header:
                sections.setdefault(header, []).append(i)
        for header, indices in sections.items():
            for i in indices:
                for j in indices:
                    if i < j:
                        self.graph.add_edge(i, j)
        logger.info("Built graph index with %d nodes.", self.graph.number_of_nodes())

    def _chunk_documents(self, documents):
        """
        Split each document into smaller chunks while preserving metadata.
        """
        return semantically_chunk_documents(documents,ollama_model=self.embedding_model)

    def _embed_and_store_chunks(self, documents):
        """
        For each document chunk, generate an embedding and store it in the vector database.
        """
        for i, doc in enumerate(documents):
            try:
                response = ollama.embeddings(model=self.embedding_model, prompt=doc.page_content)
                embedding = response.get("embedding")
                if embedding:
                    doc_id = f"chunk_{i}"
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[doc.page_content],
                        metadatas=[doc.metadata]
                    )
                else:
                    logger.error("Embedding failed for chunk %d", i)
            except Exception as e:
                logger.exception("Error embedding chunk %d: %s", i, str(e))



    def retrieve_initial_chunks(self, query: str, max_chunks: int = 10) -> list:
        """
        Retrieve candidate chunks using dense retrieval from the vector database.
        """
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=query)
            query_embedding = response.get("embedding")
            if not query_embedding:
                logger.error("Failed to generate embedding for the query.")
                return []
            results = self.collection.query(query_embeddings=[query_embedding], n_results=max_chunks)
            candidate_chunks = []
            docs_list = results.get('documents', [])
            distances_list = results.get('distances', [])
            metadatas_list = results.get('metadatas', [])
            for doc, distance, meta in zip(docs_list[0], distances_list[0], metadatas_list[0]):
                candidate_chunks.append({
                    "content": doc,
                    "distance": distance,  # lower is better
                    "metadata": meta
                })
            return candidate_chunks
        except Exception as e:
            logger.exception("Error retrieving initial (dense) chunks: %s", str(e))
            return []

    def retrieve_bm25_chunks(self, query: str, max_chunks: int = 10) -> list:
        """
        Retrieve candidate chunks using BM25-based sparse retrieval.
        """
        tokens = query.lower().split()
        scores = self.bm25_index.get_scores(tokens)
        # Get top indices sorted by BM25 score (higher is better).
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_chunks]
        candidates = []
        for i in top_indices:
            doc = self.chunked_docs[i]
            candidates.append({
                "content": doc.page_content,
                "bm25_score": scores[i],
                "metadata": doc.metadata
            })
        return candidates

    # def retrieve_graph_chunks(self, query: str, max_chunks: int = 10) -> list:
    #     """
    #     Retrieve candidate chunks from the graph database.
    #     For demonstration, perform a simple keyword matching over node content and include neighbors.
    #     """
    #     tokens = query.lower().split()
    #     candidate_nodes = []
    #     for node in self.graph.nodes(data=True):
    #         node_id, data = node
    #         content = data.get("content", "").lower()
    #         # Simple scoring: count occurrences of query tokens.
    #         score = sum(content.count(token) for token in tokens)
    #         if score > 0:
    #             candidate_nodes.append((node_id, score))
    #     candidate_nodes.sort(key=lambda x: x[1], reverse=True)
    #     results = []
    #     for node_id, score in candidate_nodes[:max_chunks]:
    #         data = self.graph.nodes[node_id]
    #         candidate = {
    #             "content": data.get("content", ""),
    #             "graph_score": score,
    #             "metadata": data.get("metadata", {})
    #         }
    #         results.append(candidate)
    #         # Include neighbors with a lower weight.
    #         for neighbor in self.graph.neighbors(node_id):
    #             neighbor_data = self.graph.nodes[neighbor]
    #             neighbor_candidate = {
    #                 "content": neighbor_data.get("content", ""),
    #                 "graph_score": score * 0.5,
    #                 "metadata": neighbor_data.get("metadata", {})
    #             }
    #             results.append(neighbor_candidate)
    #     # Remove duplicates.
    #     unique_results = []
    #     seen = set()
    #     for res in results:
    #         if res["content"] not in seen:
    #             unique_results.append(res)
    #             seen.add(res["content"])
    #     unique_results.sort(key=lambda x: x["graph_score"], reverse=True)
    #     return unique_results[:max_chunks]

    def _merge_candidates(self, dense_candidates: list, bm25_candidates: list) -> list:
        """
        Merge candidates from dense and BM25 retrieval.
        Here, we convert the dense distance to a score (using 1/(distance+ε)) and average it with the BM25 score.
        """
        merged = {}
        # Process dense candidates.
        for candidate in dense_candidates:
            # Invert the distance into a similarity score.
            dense_score = 1.0 / (candidate.get("distance", 1.0) + 1e-6)
            key = candidate["content"]
            merged[key] = {"content": candidate["content"],
                           "metadata": candidate["metadata"],
                           "dense_score": dense_score,
                           "bm25_score": 0}
        # Process BM25 candidates.
        for candidate in bm25_candidates:
            key = candidate["content"]
            if key in merged:
                merged[key]["bm25_score"] = candidate.get("bm25_score", 0)
            else:
                merged[key] = {"content": candidate["content"],
                               "metadata": candidate["metadata"],
                               "dense_score": 0,
                               "bm25_score": candidate.get("bm25_score", 0)}
        # Compute a combined score.
        merged_list = []
        for cand in merged.values():
            combined = (cand["dense_score"] + cand["bm25_score"]) / 2
            cand["combined_score"] = combined
            merged_list.append(cand)
        merged_list.sort(key=lambda x: x["combined_score"], reverse=True)
        return merged_list

    # def _is_graph_query(self, query: str) -> bool:
    #     """
    #     A simple heuristic to decide if the query is about relationships or structure.
    #     """
    #     graph_keywords = {"relationship", "connect", "network", "graph", "node", "edge", "link", "path", "hierarchy", "structure"}
    #     tokens = set(query.lower().split())
    #     return bool(tokens.intersection(graph_keywords))

    def route_query(self, query: str, max_chunks: int = 15) -> list:
        """
        Logical routing that selects the retrieval method(s) based on the query.
        - For queries containing graph-specific terms, use the graph retrieval.
        - Otherwise, use a hybrid retrieval that merges dense (vector) and BM25 (sparse) results.
        """
        # if self._is_graph_query(query):
        #     logger.info("Routing query to graph database retrieval.")
        #     return self.retrieve_graph_chunks(query, max_chunks=max_chunks)
        # else:
        logger.info("Routing query to hybrid (dense + BM25) retrieval.")
        dense_candidates = self.retrieve_initial_chunks(query, max_chunks=max_chunks)
        bm25_candidates = self.retrieve_bm25_chunks(query, max_chunks=max_chunks)
        return self._merge_candidates(dense_candidates, bm25_candidates)

    def llm_relevance_score(self, query: str, chunk_content: str) -> float:
        """
        Use the LLM to score the relevance of a text chunk.
        Extracts a JSON object with a "score" key from the response.
        """
        try:
            prompt = (
                "Evaluate the following document chunk for its relevance to the query.\n\n"
                "On a scale of 1 to 10 (where 1 is not at all relevant and 10 is extremely relevant), "
                "please provide an explanation and then output a JSON object containing only one key 'score' with your rating.\n\n"
                f"Query: {query}\n"
                f"Text: {chunk_content}\n"
            )
            response = ollama.generate(model=self.relevance_generation_model, prompt=prompt)
            score_text = response.get("response", "").strip()
            logger.info("Raw relevance score response: %s", score_text)
            match = re.search(r'\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*\}', score_text)
            if match:
                score = float(match.group(1))
                return score
            else:
                raise ValueError("No valid JSON object with 'score' key found.")
        except Exception as e:
            logger.exception("Error obtaining LLM relevance score: %s", str(e))
            return 5.0  # Fallback neutral score

    def rerank_chunks(self, query: str, candidate_chunks: list) -> list:
        """
        Re-rank candidate chunks using LLM-based relevance scoring.
        """
        reranked = []
        for chunk in candidate_chunks:
            score = self.llm_relevance_score(query, chunk["content"])
            chunk["llm_score"] = score
            reranked.append(chunk)
        reranked.sort(key=lambda x: x["llm_score"], reverse=True)
        logger.info("Re-ranked %d chunks using LLM relevance scoring.", len(reranked))
        return reranked

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in the text. Replace this simple word count with your model's tokenizer if available.
        """
        # If using tiktoken:
        # return len(tokenizer.encode(text))
        return len(text.split())

    def assemble_context(self, ranked_chunks: list, token_budget: int) -> list:
        """
        Assemble context chunks without exceeding the token budget.
        """
        context = []
        total_tokens = 0
        for chunk in ranked_chunks:
            chunk_tokens = self._count_tokens(chunk["content"])
            if total_tokens + chunk_tokens <= token_budget:
                context.append(chunk)
                total_tokens += chunk_tokens
            else:
                remaining_tokens = token_budget - total_tokens
                if remaining_tokens > 0:
                    truncated = " ".join(chunk["content"].split()[:remaining_tokens])
                    context.append({
                        "content": truncated,
                        "llm_score": chunk.get("llm_score", 0),
                        "metadata": chunk["metadata"]
                    })
                    total_tokens += remaining_tokens
                break
        logger.info("Assembled context with ~%d tokens.", total_tokens)
        return context

    def iterative_context_expansion(self, query: str, current_context: list, token_budget: int, max_rounds: int = 3) -> list:
        """
        Iteratively expand the context if it does not fill enough of the token budget.
        """
        round_counter = 0
        while round_counter < max_rounds:
            current_token_count = sum(self._count_tokens(chunk["content"]) for chunk in current_context)
            if current_token_count >= 0.7 * token_budget:
                break

            current_context_text = "\n\n".join(chunk["content"] for chunk in current_context)
            expansion_prompt = (
                "The current context for answering the query is provided below. "
                "Identify additional keywords or topics that could provide more relevant information.\n\n"
                f"Current context:\n{current_context_text}\n\n"
                "List additional keywords or phrases (comma-separated) to expand the search:"
            )
            try:
                expansion_response = ollama.generate(model=self.generation_model, prompt=expansion_prompt)
                keywords = expansion_response.get("response", "").strip()
                logger.info("Round %d expansion keywords: %s", round_counter + 1, keywords)
            except Exception as e:
                logger.exception("Error during iterative context expansion: %s", str(e))
                break

            if not keywords:
                break

            refined_query = f"{query} {keywords}"
            additional_candidates = self.route_query(refined_query, max_chunks=5)
            additional_candidates = self.rerank_chunks(query, additional_candidates)
            existing_contents = {chunk["content"] for chunk in current_context}
            for candidate in additional_candidates:
                if candidate["content"] not in existing_contents:
                    current_context.append(candidate)
                    existing_contents.add(candidate["content"])
                    current_token_count += self._count_tokens(candidate["content"])
                    if current_token_count >= token_budget:
                        break
            round_counter += 1
        return current_context

    def generate_response(self, query: str, context_chunks: list) -> str:
        """
        Generate the final LLM response using the aggregated context.
        Implements a Fusion-in-Decoder style prompt.
        """
        evidence_blocks = "\n\n".join(
            f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
            for chunk in context_chunks
        )
        full_prompt = get_llm_json_instructions(query, evidence_blocks)
        logger.info("Final prompt provided to LLM:\n%s", full_prompt)
        try:
            response = ollama.generate(model=self.generation_model, prompt=full_prompt)
            generated_text = response.get("response", "No response generated.").strip()
            logger.info("Raw generated response: %s", generated_text)
            json_matches = re.findall(r'\{[^}]*\}', generated_text)
            for json_str in json_matches:
                try:
                    result_obj = json.loads(json_str)
                    if "answer" in result_obj:
                        answer = result_obj["answer"]
                        if isinstance(answer, list):
                            return ", ".join(answer)
                        return answer
                except json.JSONDecodeError:
                    continue
            raise ValueError("No valid JSON with 'answer' key found.")
        except Exception as e:
            logger.exception("Error generating final response: %s", str(e))
            return "Error generating response."

    def query(self, query: str, max_chunks: int = 15, token_budget: int = 1024) -> str:
        """
        High-level method to process a query:
          1. Retrieve candidate chunks using logical routing.
          2. Re-rank them with LLM-based relevance scoring.
          3. Assemble the context within the token budget.
          4. Iteratively expand the context if needed.
          5. Generate the final answer using a Fusion-in-Decoder prompt.
        """
        # Use logical routing to get initial candidates.
        candidates = self.route_query(query, max_chunks=max_chunks)
        if not candidates:
            logger.warning("No candidate chunks retrieved for query: %s", query)
            return "No relevant documents found."
        ranked_candidates = self.rerank_chunks(query, candidates)
        context_chunks = self.assemble_context(ranked_candidates, token_budget)
        expanded_context = self.iterative_context_expansion(query, context_chunks, token_budget)
        return self.generate_response(query, expanded_context)


# Singleton instance for the engine.
_engine_instance = None


def init_engine(json_file_path: str, **kwargs) -> LLMQueryEngine:
    """
    Initialize the LLMQueryEngine.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMQueryEngine(json_file_path, **kwargs)
    return _engine_instance


def query_llm(query: str, max_chunks: int = 15, token_budget: int = 1024) -> str:
    """
    Public function to query the LLM engine.
    """
    global _engine_instance
    if _engine_instance is None:
        raise Exception("Engine not initialized. Please call init_engine(json_file_path) first.")
    return _engine_instance.query(query, max_chunks=max_chunks, token_budget=token_budget)


# (Optional) Run the script standalone.
if __name__ == "__main__":
    json_file_path = "data/alexnet/doc_alexnet.json"
    engine = init_engine(
        json_file_path,
        collection_name="nn_docs",
        chunk_size=1000,
        chunk_overlap=200
    )

    queries = [
        ("""Name each layer of this neural network sequentially. Do not generalize internal layers and include modification and activation layers""", 1024),
        # ("""What relationships exist between the convolutional layers and the pooling layers in this network?""", 1024)
    ]

    for idx, (q, budget) in enumerate(queries, start=1):
        start_time = time.time()
        answer = query_llm(q, token_budget=budget)
        elapsed = time.time() - start_time
        print(f"Example {idx}:\nAnswer: {answer}\nTime: {elapsed:.2f} seconds\n")



# #!/usr/bin/env python3
# """
# llm_service.py

# A Retrieval-Augmented Generation (RAG) pipeline that incorporates state-of-the-art methods:
#   - Two-stage retrieval (dense vector search, with optional hybrid/sparse methods) and LLM-based re-ranking.
#   - Fusion-in-Decoder style context assembly: each retrieved chunk is marked with its metadata so that the LLM fuses evidence from multiple sources.
#   - Iterative retrieval expansion via LLM feedback: if the initial context does not fill the token budget, the LLM suggests additional keywords to refine the query.
#   - End-to-end modular design for potential integration of interleaved retrieval or end-to-end differentiable methods.
  
# Dependencies include:
#   - `ollama` for LLM calls,
#   - `chromadb` for the vector database,
#   - LangChain’s Document abstraction and RecursiveCharacterTextSplitter,
#   - A custom JSON document loader (`load_documents_from_json`).

# Adjust model names, API calls, and error handling as needed for your environment.
# """

# import json
# import logging
# import time
# import re


# import ollama
# import chromadb

# from langchain_core.documents.base import Document
# from utils.doc_chunker import semantically_chunk_documents
# from utils.document_json_utils import load_documents_from_json

# # Configure logging.
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def get_llm_json_instructions(query: str, context: str) -> str:
#     """
#     Build a Fusion-in-Decoder style prompt that instructs the LLM to consider each evidence block.
    
#     The prompt:
#       - Provides a clear instruction to answer the query.
#       - Includes context chunks (each preceded by a header) as separate evidence sections.
#       - Instructs the LLM to output its answer as a JSON object (with key "answer").
#     """
#     instructions = (
#         "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
#         "Below are several context sections, each starting with a header. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
#         "### Evidence Blocks:\n"
#         f"{context}\n\n"
#         f"Query: {query}\n"
#         "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
#         "with your answer listed as the value. For example, the last line of your output should be:\n"
#         """ {"answer": ["name1","name2","name3"]}"""
#     )
#     return instructions


# class LLMQueryEngine:
#     """
#     A LLM engine that implements a multi-stage retrieval-augmented generation pipeline:
#       1. Loads and chunks documents (preserving metadata such as headers).
#       2. Embeds and stores chunks in a vector database (using dense retrieval; optional hybrid retrieval may be added).
#       3. Retrieves candidate chunks via similarity search.
#       4. Re-ranks candidates using an LLM to assess relevance.
#       5. Iteratively expands context if the initial retrieval does not fill the token budget.
#       6. Generates a final response using a Fusion-in-Decoder style prompt.
#     """

#     def __init__(self,
#                  json_file_path: str,
#                  collection_name: str = "nn_docs",
#                  chunk_size: int = 1000,
#                  chunk_overlap: int = 200,
#                  embedding_model: str = "bge-m3",
#                  generation_model: str = "deepseek-r1:32b",
#                  relevance_generation_model: str = "deepseek-r1:8b"):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.embedding_model = embedding_model
#         self.generation_model = generation_model
#         self.relevance_generation_model = relevance_generation_model

#         # Initialize the vector database.
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection(name=collection_name)
#         logger.info("Initialized vector DB collection: %s", collection_name)

#         # Load documents from JSON and process them.
#         docs = load_documents_from_json(json_file_path)
#         logger.info("Loaded %d documents from %s", len(docs), json_file_path)

#         self.chunked_docs = self._chunk_documents(docs)
#         logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

#         self._embed_and_store_chunks(self.chunked_docs)
#         logger.info("Finished embedding and storing document chunks.")

#     def _chunk_documents(self, documents):
#         """
#         Split each document into smaller chunks while preserving metadata (e.g. section headers).
#         """
#         return semantically_chunk_documents(documents)

#     def _embed_and_store_chunks(self, documents):
#         """
#         For each document chunk, generate an embedding and store it in the vector database.
#         """
#         for i, doc in enumerate(documents):
#             try:
#                 response = ollama.embeddings(model=self.embedding_model, prompt=doc.page_content)
#                 embedding = response.get("embedding")
#                 if embedding:
#                     doc_id = f"chunk_{i}"
#                     self.collection.add(
#                         ids=[doc_id],
#                         embeddings=[embedding],
#                         documents=[doc.page_content],
#                         metadatas=[doc.metadata]
#                     )
#                 else:
#                     logger.error("Embedding failed for chunk %d", i)
#             except Exception as e:
#                 logger.exception("Error embedding chunk %d: %s", i, str(e))

#     def retrieve_initial_chunks(self, query: str, max_chunks: int = 10) -> list:
#         """
#         Retrieve candidate chunks from the vector DB based on the query.
#         This uses dense retrieval via vector search. (Optionally, a sparse or hybrid method can be integrated.)
#         """
#         try:
#             response = ollama.embeddings(model=self.embedding_model, prompt=query)
#             query_embedding = response.get("embedding")
#             if not query_embedding:
#                 logger.error("Failed to generate embedding for the query.")
#                 return []
#             results = self.collection.query(query_embeddings=[query_embedding], n_results=max_chunks)
#             candidate_chunks = []
#             # Assuming results returns lists under these keys.
#             docs_list = results.get('documents', [])
#             distances_list = results.get('distances', [])
#             metadatas_list = results.get('metadatas', [])
#             for doc, distance, meta in zip(docs_list[0], distances_list[0], metadatas_list[0]):
#                 candidate_chunks.append({
#                     "content": doc,
#                     "distance": distance,
#                     "metadata": meta
#                 })
#             return candidate_chunks
#         except Exception as e:
#             logger.exception("Error retrieving initial chunks: %s", str(e))
#             return []

#     def llm_relevance_score(self, query: str, chunk_content: str) -> float:
#             """
#             Use the LLM to score the relevance of a text chunk with respect to the query.
#             Extracts a JSON object containing a "score" key (value between 1 and 10) from anywhere in the response.
#             """
#             try:
#                 prompt = (
#                     "Evaluate the following document chunk for its relevance to the query.\n\n"
#                     "On a scale of 1 to 10, where 1 is not at all relevant and 10 is extremely relevant, "
#                     "please provide an explanation first, and then somewhere in your response, output a JSON object that "
#                     "contains only one key 'score' with your numerical rating as the value. "
#                     "For example, a valid output could include: {\"score\": 7.5}\n\n"
#                     f"Query: {query}\n"
#                     f"Text: {chunk_content}\n"
#                 )

#                 response = ollama.generate(model=self.relevance_generation_model, prompt=prompt)
#                 score_text = response.get("response", "").strip()
#                 logger.info("Raw relevance score response: %s", score_text)

#                 # Find JSON object in the text
#                 match = re.search(r'\{\s*"score"\s*:\s*(\d+(?:\.\d+)?)\s*\}', score_text)
#                 if match:
#                     score = float(match.group(1))
#                     return score
#                 else:
#                     raise ValueError("No valid JSON object with 'score' key found in the response.")
            
#             except Exception as e:
#                 logger.exception("Error obtaining LLM relevance score: %s", str(e))
#                 return 5.0  # Fallback to a neutral score

#     def rerank_chunks(self, query: str, candidate_chunks: list) -> list:
#         """
#         Re-rank candidate chunks using LLM-based relevance scoring.
#         Higher LLM scores indicate higher relevance.
#         """
#         reranked = []
#         for chunk in candidate_chunks:
#             score = self.llm_relevance_score(query, chunk["content"])
#             chunk["llm_score"] = score
#             reranked.append(chunk)
#         reranked.sort(key=lambda x: x["llm_score"], reverse=True)
#         logger.info("Re-ranked %d chunks using LLM relevance scoring.", len(reranked))
#         return reranked

#     def assemble_context(self, ranked_chunks: list, token_budget: int) -> list:
#         """
#         Assemble context chunks without exceeding the token budget.
#         Uses a simple word–count heuristic.
#         """
#         context = []
#         total_tokens = 0
#         for chunk in ranked_chunks:
#             chunk_tokens = len(chunk["content"].split())
#             if total_tokens + chunk_tokens <= token_budget:
#                 context.append(chunk)
#                 total_tokens += chunk_tokens
#             else:
#                 remaining_tokens = token_budget - total_tokens
#                 if remaining_tokens > 0:
#                     truncated = " ".join(chunk["content"].split()[:remaining_tokens])
#                     context.append({
#                         "content": truncated,
#                         "llm_score": chunk.get("llm_score", 0),
#                         "metadata": chunk["metadata"]
#                     })
#                     total_tokens += remaining_tokens
#                 break
#         logger.info("Assembled context with an estimated %d tokens.", total_tokens)
#         return context

#     def iterative_context_expansion(self, query: str, current_context: list, token_budget: int) -> list:
#         """
#         If the current context does not fill a sufficient portion of the token budget, use the LLM to propose additional keywords.
#         Then re-query for more candidate chunks based on an expanded query.
#         """
#         current_token_count = sum(len(chunk["content"].split()) for chunk in current_context)
#         if current_token_count >= 0.7 * token_budget:
#             return current_context

#         # Prepare a summary of current context to guide the LLM.
#         current_context_text = "\n\n".join(chunk["content"] for chunk in current_context)
#         expansion_prompt = (
#             "The current context for answering the query is given below. "
#             "Identify additional keywords or topics that might provide more relevant information.\n\n"
#             f"Current context:\n{current_context_text}\n\n"
#             "List additional keywords or phrases (comma-separated) that could be used to expand the search:"
#         )
#         try:
#             expansion_response = ollama.generate(model=self.generation_model, prompt=expansion_prompt)
#             keywords = expansion_response.get("response", "").strip()
#             logger.info("Iterative expansion keywords: %s", keywords)
#         except Exception as e:
#             logger.exception("Error during iterative context expansion: %s", str(e))
#             keywords = ""

#         if keywords:
#             # Refine the query by appending the new keywords.
#             refined_query = f"{query} {keywords}"
#             additional_candidates = self.retrieve_initial_chunks(refined_query, max_chunks=5)
#             additional_candidates = self.rerank_chunks(query, additional_candidates)
#             # Merge additional candidates, avoiding duplicates.
#             existing_contents = {chunk["content"] for chunk in current_context}
#             for candidate in additional_candidates:
#                 if candidate["content"] not in existing_contents:
#                     current_context.append(candidate)
#                     existing_contents.add(candidate["content"])
#                     current_token_count += len(candidate["content"].split())
#                     if current_token_count >= token_budget:
#                         break
#         return current_context

#     def generate_response(self, query: str, context_chunks: list) -> str:
#         """
#         Generate the final LLM response using the aggregated context.
#         This method implements a Fusion-in-Decoder style prompt.
#         """
#         # Build evidence blocks with headers.
#         evidence_blocks = "\n\n".join(
#             f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
#             for chunk in context_chunks
#         )
#         full_prompt = get_llm_json_instructions(query, evidence_blocks)
#         logger.info("Final prompt provided to LLM:\n%s", full_prompt)
#         try:
#             response = ollama.generate(model=self.generation_model, prompt=full_prompt)
#             generated_text = response.get("response", "No response generated.").strip()
#             logger.info("Raw generated response: %s", generated_text)

#             # Find JSON objects in the response
#             json_matches = re.findall(r'\{[^}]*\}', generated_text)
#             for json_str in json_matches:
#                 try:
#                     result_obj = json.loads(json_str)
#                     if "answer" in result_obj:
#                         answer = result_obj["answer"]
#                         if isinstance(answer, list):  # If multiple answers are provided
#                             return ", ".join(answer)  # Or return as a list if desired
#                         return answer  # Single answer case
#                 except json.JSONDecodeError:
#                     continue

#             raise ValueError("No valid JSON object with 'answer' key found in the response.")
#         except Exception as e:
#             logger.exception("Error generating final response: %s", str(e))
#             return "Error generating response."


#     def query(self, query: str, max_chunks: int = 15, token_budget: int = 1024) -> str:
#         """
#         High-level method to process a query:
#           1. Retrieve candidate chunks using dense (or hybrid) retrieval.
#           2. Re-rank them with LLM-based relevance scoring.
#           3. Assemble the context within the token budget.
#           4. Optionally iteratively expand the context.
#           5. Generate the final answer with a Fusion-in-Decoder style prompt.
#         """
#         # Step 1: Retrieve initial candidates.
#         candidates = self.retrieve_initial_chunks(query, max_chunks=max_chunks)
#         if not candidates:
#             logger.warning("No candidate chunks retrieved for query: %s", query)
#             return "No relevant documents found."

#         # Step 2: Re-rank candidates.
#         ranked_candidates = self.rerank_chunks(query, candidates)
#         # Step 3: Assemble context.
#         context_chunks = self.assemble_context(ranked_candidates, token_budget)
#         # Step 4: Iteratively expand context if necessary.
#         expanded_context = self.iterative_context_expansion(query, context_chunks, token_budget)
#         # Step 5: Generate the final response.
#         return self.generate_response(query, expanded_context)


# # Singleton instance for the engine.
# _engine_instance = None


# def init_engine(json_file_path: str, **kwargs) -> LLMQueryEngine:
#     """
#     Initialize the LLMQueryEngine. Keyword arguments can include:
#       - collection_name, chunk_size, chunk_overlap, embedding_model, generation_model, etc.
#     """
#     global _engine_instance
#     if _engine_instance is None:
#         _engine_instance = LLMQueryEngine(json_file_path, **kwargs)
#     return _engine_instance


# def query_llm(query: str, max_chunks: int = 15, token_budget: int = 1024) -> str:
#     """
#     Public function to query the LLM engine.
#     Ensure that init_engine(json_file_path) has been called beforehand.
#     """
#     global _engine_instance
#     if _engine_instance is None:
#         raise Exception("Engine not initialized. Please call init_engine(json_file_path) first.")
#     return _engine_instance.query(query, max_chunks=max_chunks, token_budget=token_budget)


# # (Optional) to run the script standalone
# if __name__ == "__main__":
#     json_file_path = "data/alexnet/doc_alexnet.json"
#     # json_file_path = "data/resnet/doc_resnet.json"
#     engine = init_engine(
#         json_file_path,
#         collection_name="nn_docs",
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     # Example queries.
#     queries = [
#         ("""Name each layer of this neural network sequentially. Do not generalize internal layers and include modifiction and activationw layers""", 1024)
#         # ("What is the number of parameters in the first convolutional layer?", 512),
#         # ("What activation function is used on the second fully-connected layer, if any?", 512),
#         # ("""List the modification layers applied to the first convolutional layer, if any. 
#         #     Modification layers are integral components in neural network architectures that purposefully alter the data flow to optimize the learning process, manage complexity, and enhance robustness (e.g., Noise layers (Gaussian), Batch Normalization layers, flatten layers, separation layers (Clone or split layers)).""", 512),
#         # ("What is the name of the dataset(s) used for this model?", 512)
#     ]

#     for idx, (q, budget) in enumerate(queries, start=1):
#         start_time = time.time()
#         answer = query_llm(q, token_budget=budget)
#         elapsed = time.time() - start_time
#         print(f"Example {idx}:\nAnswer: {answer}\nTime: {elapsed:.2f} seconds\n")