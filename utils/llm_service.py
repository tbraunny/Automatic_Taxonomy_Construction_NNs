
#!/usr/bin/env python3
"""
llm_service.py

A Retrieval-Augmented Generation (RAG) pipeline that incorporates state-of-the-art methods:
  - Two-stage retrieval (dense vector search, with optional hybrid/sparse methods) and LLM-based re-ranking.
  - Fusion-in-Decoder style context assembly: each retrieved chunk is marked with its metadata so that the LLM fuses evidence from multiple sources.
  - Iterative retrieval expansion via LLM feedback: if the initial context does not fill the token budget, the LLM suggests additional keywords to refine the query.
  - End-to-end modular design for potential integration of interleaved retrieval or end-to-end differentiable methods.
  
Dependencies include:
  - `ollama` for LLM calls,
  - `chromadb` for the vector database,
  - LangChain’s Document abstraction and RecursiveCharacterTextSplitter,
  - A custom JSON document loader (`load_documents_from_json`).

Adjust model names, API calls, and error handling as needed for your environment.
"""

import json
import logging
import time

import ollama
import chromadb

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.document_json_utils import load_documents_from_json

# Configure logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_llm_json_instructions(query: str, context: str) -> str:
    """
    Build a Fusion-in-Decoder style prompt that instructs the LLM to consider each evidence block.
    
    The prompt:
      - Provides a clear instruction to answer the query.
      - Includes context chunks (each preceded by a header) as separate evidence sections.
      - Instructs the LLM to output its answer as a JSON object (with key "answer").
    """
    instructions = (
        "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
        "Below are several context sections, each starting with a header. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
        "### Evidence Blocks:\n"
        f"{context}\n\n"
        f"Query: {query}\n"
        "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
        "with your answer listed as the value. For example, the last line of your output should be:\n"
        """ {"answer": ["name1","name2","name3"]}"""
    )
    return instructions


class LLMQueryEngine:
    """
    A LLM engine that implements a multi-stage retrieval-augmented generation pipeline:
      1. Loads and chunks documents (preserving metadata such as headers).
      2. Embeds and stores chunks in a vector database (using dense retrieval; optional hybrid retrieval may be added).
      3. Retrieves candidate chunks via similarity search.
      4. Re-ranks candidates using an LLM to assess relevance.
      5. Iteratively expands context if the initial retrieval does not fill the token budget.
      6. Generates a final response using a Fusion-in-Decoder style prompt.
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

        # Load documents from JSON and process them.
        docs = load_documents_from_json(json_file_path)
        logger.info("Loaded %d documents from %s", len(docs), json_file_path)

        self.chunked_docs = self._chunk_documents(docs)
        logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

        self._embed_and_store_chunks(self.chunked_docs)
        logger.info("Finished embedding and storing document chunks.")

    def _chunk_documents(self, documents):
        """
        Split each document into smaller chunks while preserving metadata (e.g. section headers).
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunked_docs = []
        for doc in documents:
            original_metadata = doc.metadata.copy() if doc.metadata else {}
            header = original_metadata.get("section_header", "No Header")
            chunks = text_splitter.split_text(doc.page_content)
            for chunk_text in chunks:
                new_doc = Document(page_content=chunk_text,
                                   metadata={"section_header": header})
                chunked_docs.append(new_doc)
        return chunked_docs

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
        Retrieve candidate chunks from the vector DB based on the query.
        This uses dense retrieval via vector search. (Optionally, a sparse or hybrid method can be integrated.)
        """
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=query)
            query_embedding = response.get("embedding")
            if not query_embedding:
                logger.error("Failed to generate embedding for the query.")
                return []
            results = self.collection.query(query_embeddings=[query_embedding], n_results=max_chunks)
            candidate_chunks = []
            # Assuming results returns lists under these keys.
            docs_list = results.get('documents', [])
            distances_list = results.get('distances', [])
            metadatas_list = results.get('metadatas', [])
            for doc, distance, meta in zip(docs_list[0], distances_list[0], metadatas_list[0]):
                candidate_chunks.append({
                    "content": doc,
                    "distance": distance,
                    "metadata": meta
                })
            return candidate_chunks
        except Exception as e:
            logger.exception("Error retrieving initial chunks: %s", str(e))
            return []

    def llm_relevance_score(self, query: str, chunk_content: str) -> float:
        """
        Use the LLM to score the relevance of a text chunk with respect to the query.
        Instructs the LLM to output a JSON object with a "score" key (value between 1 and 10).
        """
        try:
            # prompt = (
            #     "Evaluate the following document chunk for its relevance to the query.\n\n"
            #     f"Query: {query}\n\n"
            #     f"Document chunk: {chunk_content}\n\n"
            #     "Please provide an explanation first, and then on a new line, output a JSON object that contains only one key 'score' "
            #     "with your numerical rating as the value. For example, the last line of your output should be:\n"
            #     "{\"score\": 7.5}\n\n"
            #     "On a scale from 1 (not relevant) to 10 (highly relevant), please output a JSON object on a new line "
            #     "with a single key 'score'."
            # )
            prompt = (
                "Evaluate the following document chunk for its relevance to the query.\n\n"
                "On a scale of 1 to 10, where 1 is not at all relevant and 10 is extremely relevant, "
                "Please provide an explanation first, and then on a new line, output a JSON object that contains only one key 'score' "
                "with your numerical rating as the value. For example, the last line of your output should be:\n"
                "{\"score\": 7.5}\n\n"
                f"Query: {query}\n"
                f"Text: {chunk_content}\n"
            )
            # Using a smaller model for scoring (adjust as needed).
            response = ollama.generate(model=self.relevance_generation_model, prompt=prompt)
            score_text = response.get("response", "").strip()
            logger.info("Raw relevance score response: %s", score_text)

            # Assume the final non-empty line is a JSON object.
            lines = [line.strip() for line in score_text.splitlines() if line.strip()]
            if not lines:
                raise ValueError("No output lines found in the LLM response.")
            json_line = lines[-1]
            logger.info("Extracted JSON line for score: %s", json_line)
            score_obj = json.loads(json_line)
            if "score" in score_obj:
                return float(score_obj["score"])
            else:
                raise ValueError(f"'score' key not found in JSON: {json_line}")
        except Exception as e:
            logger.exception("Error obtaining LLM relevance score: %s", str(e))
            # Fallback to a neutral score.
            return 5.0

    def rerank_chunks(self, query: str, candidate_chunks: list) -> list:
        """
        Re-rank candidate chunks using LLM-based relevance scoring.
        Higher LLM scores indicate higher relevance.
        """
        reranked = []
        for chunk in candidate_chunks:
            score = self.llm_relevance_score(query, chunk["content"])
            chunk["llm_score"] = score
            reranked.append(chunk)
        reranked.sort(key=lambda x: x["llm_score"], reverse=True)
        logger.info("Re-ranked %d chunks using LLM relevance scoring.", len(reranked))
        return reranked

    def assemble_context(self, ranked_chunks: list, token_budget: int) -> list:
        """
        Assemble context chunks without exceeding the token budget.
        Uses a simple word–count heuristic.
        """
        context = []
        total_tokens = 0
        for chunk in ranked_chunks:
            chunk_tokens = len(chunk["content"].split())
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
        logger.info("Assembled context with an estimated %d tokens.", total_tokens)
        return context

    def iterative_context_expansion(self, query: str, current_context: list, token_budget: int) -> list:
        """
        If the current context does not fill a sufficient portion of the token budget, use the LLM to propose additional keywords.
        Then re-query for more candidate chunks based on an expanded query.
        """
        current_token_count = sum(len(chunk["content"].split()) for chunk in current_context)
        if current_token_count >= 0.7 * token_budget:
            return current_context

        # Prepare a summary of current context to guide the LLM.
        current_context_text = "\n\n".join(chunk["content"] for chunk in current_context)
        expansion_prompt = (
            "The current context for answering the query is given below. "
            "Identify additional keywords or topics that might provide more relevant information.\n\n"
            f"Current context:\n{current_context_text}\n\n"
            "List additional keywords or phrases (comma-separated) that could be used to expand the search:"
        )
        try:
            expansion_response = ollama.generate(model=self.generation_model, prompt=expansion_prompt)
            keywords = expansion_response.get("response", "").strip()
            logger.info("Iterative expansion keywords: %s", keywords)
        except Exception as e:
            logger.exception("Error during iterative context expansion: %s", str(e))
            keywords = ""

        if keywords:
            # Refine the query by appending the new keywords.
            refined_query = f"{query} {keywords}"
            additional_candidates = self.retrieve_initial_chunks(refined_query, max_chunks=5)
            additional_candidates = self.rerank_chunks(query, additional_candidates)
            # Merge additional candidates, avoiding duplicates.
            existing_contents = {chunk["content"] for chunk in current_context}
            for candidate in additional_candidates:
                if candidate["content"] not in existing_contents:
                    current_context.append(candidate)
                    existing_contents.add(candidate["content"])
                    current_token_count += len(candidate["content"].split())
                    if current_token_count >= token_budget:
                        break
        return current_context

    def generate_response(self, query: str, context_chunks: list) -> str:
        """
        Generate the final LLM response using the aggregated context.
        This method implements a Fusion-in-Decoder style prompt.
        """
        # Build evidence blocks with headers.
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
            # Expect the last non-empty line to be JSON.
            lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
            if not lines:
                raise ValueError("No output lines found in the LLM response.")
            json_line = lines[-1]
            logger.info("Extracted JSON line: %s", json_line)
            result_obj = json.loads(json_line)
            if "answer" in result_obj:
                answer = result_obj["answer"]
                if isinstance(answer, list):  # If multiple answers are provided
                    return ", ".join(answer)  # Or return as a list if desired
                return answer  # Single answer case
            else:
                raise ValueError(f"'answer' key not found in JSON: {json_line}")
        except Exception as e:
            logger.exception("Error generating final response: %s", str(e))
            return "Error generating response."

    def query(self, query: str, max_chunks: int = 10, token_budget: int = 1024) -> str:
        """
        High-level method to process a query:
          1. Retrieve candidate chunks using dense (or hybrid) retrieval.
          2. Re-rank them with LLM-based relevance scoring.
          3. Assemble the context within the token budget.
          4. Optionally iteratively expand the context.
          5. Generate the final answer with a Fusion-in-Decoder style prompt.
        """
        # Step 1: Retrieve initial candidates.
        candidates = self.retrieve_initial_chunks(query, max_chunks=max_chunks)
        if not candidates:
            logger.warning("No candidate chunks retrieved for query: %s", query)
            return "No relevant documents found."

        # Step 2: Re-rank candidates.
        ranked_candidates = self.rerank_chunks(query, candidates)
        # Step 3: Assemble context.
        context_chunks = self.assemble_context(ranked_candidates, token_budget)
        # Step 4: Iteratively expand context if necessary.
        expanded_context = self.iterative_context_expansion(query, context_chunks, token_budget)
        # Step 5: Generate the final response.
        return self.generate_response(query, expanded_context)


# Singleton instance for the engine.
_engine_instance = None


def init_engine(json_file_path: str, **kwargs) -> LLMQueryEngine:
    """
    Initialize the LLMQueryEngine. Keyword arguments can include:
      - collection_name, chunk_size, chunk_overlap, embedding_model, generation_model, etc.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMQueryEngine(json_file_path, **kwargs)
    return _engine_instance


def query_llm(query: str, max_chunks: int = 10, token_budget: int = 1024) -> str:
    """
    Public function to query the LLM engine.
    Ensure that init_engine(json_file_path) has been called beforehand.
    """
    global _engine_instance
    if _engine_instance is None:
        raise Exception("Engine not initialized. Please call init_engine(json_file_path) first.")
    return _engine_instance.query(query, max_chunks=max_chunks, token_budget=token_budget)


if __name__ == "__main__":
    json_file_path = "data/alexnet/doc_alexnet.json"
    engine = init_engine(
        json_file_path,
        collection_name="nn_docs",
        chunk_size=1000,
        chunk_overlap=200
    )

    # Example queries.
    queries = [
        ("What is the number of parameters in the first convolutional layer?", 512),
        ("What activation function is used on the second fully-connected layer, if any?", 512),
        ("""List the modification layers applied to the first convolutional layer, if any. 
            Modification layers are integral components in neural network architectures that purposefully alter the data flow to optimize the learning process, manage complexity, and enhance robustness (e.g., Noise layers (Gaussian), Batch Normalization layers, flatten layers, separation layers (Clone or split layers)).""", 512),
        ("What is the name of the dataset(s) used for this model?", 512)
    ]

    for idx, (q, budget) in enumerate(queries, start=1):
        start_time = time.time()
        answer = query_llm(q, token_budget=budget)
        elapsed = time.time() - start_time
        print(f"Example {idx}:\nAnswer: {answer}\nTime: {elapsed:.2f} seconds\n")


# """
# robust_llm_service.py

# A robust Retrieval-Augmented Generation (RAG) pipeline that goes beyond pure similarity-based retrieval.
# In addition to a vector database search, this engine:
#   - Re–ranks candidate document chunks using an LLM.
#   - Iteratively expands the retrieved context using LLM feedback.
#   - Dynamically assembles context (with truncation or expansion) so that the LLM always sees the best information.

# Note: This script uses:
#     - `ollama` as the LLM service,
#     - `chromadb` as the vector database,
#     - LangChain’s Document abstraction and a recursive text splitter,
#     - A custom JSON document loader (assumed available as load_documents_from_json).
# You may need to adjust model names, API calls, and error handling to your setup.
# """

# import logging
# import time

# import ollama
# import chromadb
# import json

# from langchain_core.documents.base import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils.document_json_utils import load_documents_from_json

# # Configure logging.
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def get_llm_json_instructions(query, context):
#     """
#     Construct the LLM prompt.
#     Answers the query based on context.
#     """
#     instructions = (
#         "Using the provided context, answer the following question concisely and accurately. "
#         "If the answer is not clearly supported by the context, state that the context is insufficient.\n\n"
#         f"Context:\n{context}\n\n"
#         f"Question: {query}\n"
#         "Please provide an explanation first, and then on a new line, output a JSON object that contains only one key 'answer' "
#         "with your answer listed as the value. For example, the last line of your output should be:\n"
#         "{\"answer\": name1, name2, name3}\n\n"
#         )
#     return instructions


# class RobustLLMQueryEngine:
#     """
#     A robust LLM engine that:
#       - Loads and chunks documents (while preserving headers).
#       - Embeds and stores chunks in a vector database.
#       - Retrieves candidate chunks via similarity search.
#       - Re-ranks candidates using an LLM.
#       - Iteratively expands the context when needed.
#       - Generates a final response based on an assembled context.
#     """

#     def __init__(self,
#                  json_file_path,
#                  collection_name="robust_docs",
#                  chunk_size=1000,
#                  chunk_overlap=200,
#                 #  embedding_model="mxbai-embed-large:latest",
#                  embedding_model="bge-m3",

#                  generation_model="deepseek-r1:32b"):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.embedding_model = embedding_model
#         self.generation_model = generation_model

#         # Initialize the vector database.
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection(name=collection_name)
#         logger.info("Initialized robust vector DB collection: %s", collection_name)

#         # Load documents from JSON and process them.
#         docs = load_documents_from_json(json_file_path)
#         logger.info("Loaded %d documents from %s", len(docs), json_file_path)

#         self.chunked_docs = self._chunk_documents(docs)
#         logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

#         self._embed_and_store_chunks(self.chunked_docs)
#         logger.info("Finished embedding and storing document chunks.")

#     def _chunk_documents(self, documents):
#         """
#         Split each document into smaller chunks while preserving metadata (e.g., section headers).
#         """
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#         )
#         chunked_docs = []
#         for doc in documents:
#             original_metadata = doc.metadata.copy() if doc.metadata else {}
#             header = original_metadata.get("section_header", "No Header")
#             chunks = text_splitter.split_text(doc.page_content)
#             for chunk_text in chunks:
#                 new_doc = Document(page_content=chunk_text,
#                                    metadata={"section_header": header})
#                 chunked_docs.append(new_doc)
#         return chunked_docs

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

#     def retrieve_initial_chunks(self, query, max_chunks=10, token_budget=1024):
#         """
#         Retrieve candidate chunks from the vector DB based on the query.
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




#     def llm_relevance_score(self, query, chunk_content):
#         """
#         Use the LLM to score the relevance of a text chunk with respect to the query.
#         Expects a numerical answer between 1 and 10.
#         This implementation instructs the LLM to output a JSON object with a "score" key on a new line.
#         """
#         try:
#             prompt = (
#                 "On a scale of 1 to 10, where 1 is not at all relevant and 10 is extremely relevant, "
#                 "rate the relevance of the following text in answering the query.\n\n"
#                 "Please provide an explanation first, and then on a new line, output a JSON object that contains only one key 'score' "
#                 "with your numerical rating as the value. For example, the last line of your output should be:\n"
#                 "{\"score\": 7.5}\n\n"
#                 f"Query: {query}\n"
#                 f"Text: {chunk_content}\n"
#             )
#             # Using a specific model for illustration; replace with your desired generation model.
#             response = ollama.generate(model="deepseek-r1:8b", prompt=prompt)
#             score_text = response.get("response", "").strip()
#             logger.info("Raw relevance score response: %s", score_text)

#             # Attempt to extract the JSON object from the final line.
#             # We assume that the JSON object is on the last non-empty line.
#             lines = [line.strip() for line in score_text.splitlines() if line.strip()]
#             if not lines:
#                 raise ValueError("No output lines found in the LLM response.")
            
#             json_line = lines[-1]
#             logger.info("Extracted JSON line for score: %s", json_line)
            
#             # Parse the JSON line.
#             score_obj = json.loads(json_line)
#             if "score" in score_obj:
#                 score = float(score_obj["score"])
#                 return score
#             else:
#                 raise ValueError(f"'score' key not found in JSON: {json_line}")
#         except Exception as e:
#             logger.exception("Error obtaining LLM relevance score: %s", str(e))
#             # Return a default neutral score if extraction fails.
#             return 5.0

#     def rerank_chunks(self, query, candidate_chunks):
#         """
#         Re-rank candidate chunks using LLM-based relevance scoring.
#         """
#         reranked = []
#         for chunk in candidate_chunks:
#             score = self.llm_relevance_score(query, chunk["content"])
#             chunk["llm_score"] = score
#             reranked.append(chunk)
#         # Higher scores indicate higher relevance.
#         reranked.sort(key=lambda x: x["llm_score"], reverse=True)
#         logger.info("Re-ranked %d chunks using LLM relevance scoring.", len(reranked))
#         return reranked

#     def assemble_context(self, ranked_chunks, token_budget):
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
#                 # Add as much as possible from the current chunk.
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

#     def iterative_context_expansion(self, query, current_context, token_budget):
#         """
#         If the current context does not fill a sufficient portion of the token budget,
#         ask the LLM to suggest additional keywords/topics and re-query for more chunks.
#         """
#         current_token_count = sum(len(chunk["content"].split()) for chunk in current_context)
#         # Only expand if we haven't reached ~70% of the token budget.
#         if current_token_count >= 0.7 * token_budget:
#             return current_context

#         # Combine the current context into a single text.
#         current_context_text = "\n\n".join(chunk["content"] for chunk in current_context)
#         expansion_prompt = (
#             f"Given the following context:\n{current_context_text}\n\n"
#             f"And the query: {query}\n"
#             "List additional keywords or topics that might be missing but are important to answer the query. "
#             "Provide a comma-separated list of keywords."
#         )
#         try:
#             expansion_response = ollama.generate(model=self.generation_model, prompt=expansion_prompt)
#             keywords = expansion_response.get("response", "")
#             logger.info("Iterative expansion keywords: %s", keywords)
#         except Exception as e:
#             logger.exception("Error during iterative context expansion: %s", str(e))
#             keywords = ""

#         if keywords:
#             # Form a refined query by appending the new keywords.
#             refined_query = query + " " + keywords
#             additional_candidates = self.retrieve_initial_chunks(refined_query, max_chunks=5, token_budget=token_budget)
#             additional_candidates = self.rerank_chunks(query, additional_candidates)
#             # Merge additional candidates with the current context, avoiding duplicates.
#             existing_contents = {chunk["content"] for chunk in current_context}
#             for candidate in additional_candidates:
#                 if candidate["content"] not in existing_contents:
#                     current_context.append(candidate)
#                     existing_contents.add(candidate["content"])
#                     current_token_count += len(candidate["content"].split())
#                     if current_token_count >= token_budget:
#                         break
#         return current_context

#     def generate_response(self, query, context_chunks):
#         """
#         Generate the final LLM response using the aggregated context.
#         """
#         try:
#             full_context = "\n\n".join(
#                 f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
#                 for chunk in context_chunks
#             )
#             logger.info("Final context provided to LLM:\n%s", full_context)
#             full_prompt = get_llm_json_instructions(query, full_context)
#             response = ollama.generate(model=self.generation_model, prompt=full_prompt)
#             generated_text = response.get("response", "No response generated.").strip()

#             logger.info("Raw generated query response: %s", generated_text)

#             # Attempt to extract the JSON object from the final line.
#             # We assume that the JSON object is on the last non-empty line.
#             lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
#             if not lines:
#                 raise ValueError("No output lines found in the LLM response.")
            
#             json_line = lines[-1]
#             logger.info("Extracted JSON line for generated query response: %s", json_line)
            
#             # Parse the JSON line.
#             score_obj = json.loads(json_line)
#             if "answer" in score_obj:
#                 answer = float(score_obj["answer"])
#                 return answer
#             else:
#                 raise ValueError(f"'answer' key not found in JSON: {json_line}")

#             return generated_text
#         except Exception as e:
#             logger.exception("Error generating final response: %s", str(e))
#             return "Error generating response."

#     def query(self, query, max_chunks=10, token_budget=1024):
#         """
#         High-level method to process a query:
#           1. Retrieve candidate chunks.
#           2. Re-rank them using LLM-based scoring.
#           3. Assemble the context within the token budget.
#           4. Optionally iteratively expand the context.
#           5. Generate the final answer.
#         """
#         # Step 1: Retrieve initial candidates.
#         candidates = self.retrieve_initial_chunks(query, max_chunks=max_chunks, token_budget=token_budget)
#         if not candidates:
#             logger.warning("No candidate chunks retrieved for query: %s", query)
#             return "No relevant documents found."

#         # Step 2: Re-rank candidates.
#         ranked_candidates = self.rerank_chunks(query, candidates)

#         # Step 3: Assemble context.
#         context_chunks = self.assemble_context(ranked_candidates, token_budget)

#         # Step 4: Iteratively expand context if necessary.
#         expanded_context = self.iterative_context_expansion(query, context_chunks, token_budget)

#         # Step 5: Generate and return the final response.
#         return self.generate_response(query, expanded_context)


# # Singleton instance for the engine.
# _robust_engine_instance = None


# def init_robust_engine(json_file_path, **kwargs):
#     """
#     Initialize the RobustLLMQueryEngine. Keyword arguments can include:
#       - collection_name, chunk_size, chunk_overlap, embedding_model, generation_model, etc.
#     """
#     global _robust_engine_instance
#     if _robust_engine_instance is None:
#         _robust_engine_instance = RobustLLMQueryEngine(json_file_path, **kwargs)
#     return _robust_engine_instance


# def query_robust_llm(query, max_chunks=10, token_budget=1024):
#     """
#     Public function to query the robust LLM engine.
#     Ensure that init_robust_engine(json_file_path) has been called beforehand.
#     """
#     global _robust_engine_instance
#     if _robust_engine_instance is None:
#         raise Exception("Engine not initialized. Please call init_robust_engine(json_file_path) first.")
#     return _robust_engine_instance.query(query, max_chunks=max_chunks, token_budget=token_budget)


# if __name__ == "__main__":
#     # -----------------------------
#     # Robust RAG pipeline.
#     # -----------------------------
#     json_file_path = "data/alexnet/doc_alexnet.json"
#     engine = init_robust_engine(
#         json_file_path,
#         collection_name="nn_robust_docs",
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     import time


#     start_time = time.time()
#     sample_query = "How many parameters are in the first convolutional layer?"
#     answer = query_robust_llm(sample_query, token_budget=512)
#     print("Example 1:\n", answer)
#     print("Time 1:", time.time() - start_time)



#     start_time = time.time()
#     ontology_query = "What activation function is used on the second fully-connected layer, if any."
#     ontology_answer = query_robust_llm(ontology_query, token_budget=512)
#     print("Example 2:\n", ontology_answer)
#     print("Time 2:", time.time() - start_time)



#     start_time = time.time()
#     sample_query = """List the modifiction layers applied to the first convolutional layer, if any. Modification layers are integral components in neural network architectures that purposefully alter the data flow to optimize the learning process, manage complexity, and enhance the robustness of the model (i.e. Noise layers (Gaussian), Batch Normalization layers, flatten layers, separation layers (Clone or split layers))"""
#     answer = query_robust_llm(sample_query, token_budget=512)
#     print("Example 3:\n", answer)
#     print("Time 3:", time.time() - start_time)

#     start_time = time.time()
#     sample_query = """What is the name of the dataset(s) used for this model."""
#     answer = query_robust_llm(sample_query, token_budget=512)
#     print("Example 4:\n", answer)
#     print("Time 4:", time.time() - start_time)


# # llm_service.py
# import ollama
# import chromadb

# from langchain_core.documents.base import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils.document_json_utils import load_documents_from_json


# class LLMQueryEngine:
#     """
#     An llm engine that initializes the vector database and loads/embeds documents once.
#     It provides a method to process queries using the pre-built embeddings.
#     """
#     def __init__(self, json_file_path, collection_name="file_docs", chunk_size=1000, chunk_overlap=200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap

#         # Initialize the vector database client and collection.
#         self.client = chromadb.Client()
#         self.collection = self.client.create_collection(name=collection_name)

#         # Load, chunk, embed, and store documents.
#         docs = load_documents_from_json(json_file_path)
#         split_docs = self.chunk_document_preserving_header(docs)
#         self.embed_and_store_chunks(split_docs)

#     def chunk_document_preserving_header(self, documents):
#         """
#         Splits a list of documents into smaller chunks while preserving metadata headers.
#         """
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#         )

#         chunked_docs = []
#         for doc in documents:
#             # Copy original metadata and get header (or use default).
#             original_metadata = doc.metadata.copy()
#             header = original_metadata.get("section_header", "No Header")
#             # Split the text.
#             split_chunks = text_splitter.split_text(doc.page_content)
#             for chunk_text in split_chunks:
#                 chunked_doc = Document(
#                     page_content=chunk_text,
#                     metadata={"section_header": header}
#                 )
#                 chunked_docs.append(chunked_doc)
#         return chunked_docs

#     def embed_and_store_chunks(self, documents):
#         """
#         Generates embeddings for each document chunk and stores them in the vector DB.
#         """
#         for i, doc in enumerate(documents):
#             try:
#                 response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=doc.page_content)
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
#                     raise ValueError(f"Embedding failed for chunk {i}")
#             except Exception as e:
#                 print(f"Error embedding chunk {i}: {e}")

#     def retrieve_relevant_chunks_within_token_budget(self, prompt, max_chunks=10, token_budget=1024):
#         """
#         Retrieves the most relevant document chunks given the prompt, keeping within the token budget.
#         """
#         try:
#             response = ollama.embeddings(model="mxbai-embed-large:latest", prompt=prompt)
#             embedding = response.get("embedding")
#             if not embedding:
#                 print("Error: Failed to generate embedding for prompt.")
#                 return []

#             results = self.collection.query(query_embeddings=[embedding], n_results=max_chunks)
#             # Check that the expected keys are in the results.
#             if not all(isinstance(results.get(key, []), list) for key in ['documents', 'distances', 'metadatas']):
#                 print("Unexpected response format from vector database.")
#                 return []

#             # Assemble results.
#             all_docs = []
#             for docs, scores, metas in zip(results.get('documents', []),
#                                            results.get('distances', []),
#                                            results.get('metadatas', [])):
#                 for doc, score, meta in zip(docs, scores, metas):
#                     all_docs.append({
#                         "content": doc,
#                         "score": score,
#                         "metadata": meta
#                     })

#             # Sort by score (assumes higher is better).
#             sorted_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)

#             # Build the context without exceeding the token budget.
#             context = []
#             total_tokens = 0
#             for doc in sorted_docs:
#                 # Use a rough token estimation by word count.
#                 doc_tokens = len(doc["content"].split())
#                 if total_tokens + doc_tokens <= token_budget:
#                     context.append(doc)
#                     total_tokens += doc_tokens
#                 else:
#                     break
#             return context
#         except Exception as e:
#             print(f"Error querying relevant documents: {e}")
#             return []

#     def generate_optimized_response(self, prompt, context):
#         """
#         Uses the LLM to generate a response based on the provided prompt and document context.
#         """
#         try:
#             full_context = "\n\n".join([
#                 f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
#                 for chunk in context
#             ])
#             print("### Context Provided to LLM ###")
#             print(full_context)
#             print("################################")


#             response = ollama.generate(
#                 model="deepseek-r1:32b",
#                 # prompt=(
#                 #     f"Using this context:\n{full_context}\n\n"
#                 #     f"Answer the following question concisely and accurately:\n{prompt}"
#                 # ),
#                 prompt=get_llm_json_instructions(prompt,full_context)
#                 # stream=True,
#             )
#             return response.get('response', "No response generated.")
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             return "Error in response generation."

#     def query_llm(self, prompt, max_chunks=10, token_budget=1024):
#         """
#         Public method to process a prompt:
#          - It retrieves the relevant chunks.
#          - It then passes the context and prompt to the LLM.
#         """
#         relevant_chunks = self.retrieve_relevant_chunks_within_token_budget(
#             prompt, max_chunks=max_chunks, token_budget=token_budget
#         )
#         if relevant_chunks:
#             return self.generate_optimized_response(prompt, relevant_chunks)
#         else:
#             return "No relevant documents found."


# # Singleton instance (will be initialized once)
# _engine_instance = None


# def init_engine(json_file_path, collection_name="file_docs", chunk_size=1000, chunk_overlap=200):
#     """
#     Initializes the LLMQueryEngine if not already initialized.
#     This function should be called once (for example at application startup).
#     """
#     global _engine_instance
#     if _engine_instance is None:
#         _engine_instance = LLMQueryEngine(
#             json_file_path=json_file_path,
#             collection_name=collection_name,
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap
#         )
#     return _engine_instance


# def query_llm(prompt, max_chunks=10, token_budget=512):
#     """
#     Public function that other scripts can call to get an LLM response.
#     Assumes that init_engine() has already been called.
#     """
#     global _engine_instance
#     if _engine_instance is None:
#         raise Exception("Engine not initialized. Please call init_engine(json_file_path) first.")
#     return _engine_instance.query_llm(prompt, max_chunks=max_chunks, token_budget=token_budget)


# def get_llm_json_instructions(class_prompt, rag_context):
#     return f"""
# You are an AI assistant. Solve the following problem by reasoning through it step by step. Think naturally about the problem, and at the end, provide your final response in **strict JSON format**.

# Problem: {class_prompt}

# Consider ONLY this relevant external information retrieved to help answer the question:
# [CONTEXT START]
# {rag_context}
# [CONTEXT END]

# Think through the problem step by step below:

# ---

# (Write your reasoning naturally here.)

# ---

# Once you have completed your reasoning, output **only JSON** in this format:
# ```json
# {{
#   "answer": "<your final answer here>"
# }}
# """

# # Optional: allow running this module directly for a test query.
# if __name__ == "__main__":
#     # Initialize the engine with your JSON document path.
#     json_file_path = "data/alexnet/doc_alexnet.json"
#     init_engine(json_file_path)

#     import time

#     start = time.time()

#     # Process a sample query.
#     sample_prompt = "How many parameters are in the first convolutional layer in the convolutional network, including bias terms"
#     answer = query_llm(sample_prompt)
#     print("\nFinal Answer:\n", answer)

#     end = time.time()
#     print(end - start)


#     start = time.time()
#     # Process a sample query.
#     sample_prompt = "How does the model account for overfitting"
#     answer = query_llm(sample_prompt)
#     print("\nFinal Answer:\n", answer)

#     end = time.time()
#     print(end - start)

