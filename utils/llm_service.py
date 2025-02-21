"""
Advanced RAG Pipeline with FAISS for In‐Memory Dense Retrieval

This implementation integrates several improvements:
  1. In‐memory FAISS vector index for dense retrieval (no persistence beyond runtime).
  2. BM25-based sparse retrieval.
  3. Asynchronous wrappers and batch processing.
  4. Accurate token counting via tiktoken (if available) and summarization for over‐budget chunks.
  5. Robust error handling with retries and fallbacks.
  6. Configuration via environment variables.
  7. Improved candidate merging (weighted averaging and deduplication).
  8. Dynamic, round-robin context assembly.
  9. A simple graph index (using networkx) to simulate more advanced graph-based retrieval.
 10. Detailed logging and instrumentation.
 11. A stub for user feedback submission for continuous improvement.

Dependencies include:
  - ollama (for embeddings and generation),
  - faiss and numpy for dense vector retrieval,
  - rank_bm25 for BM25 retrieval,
  - networkx,
  - tiktoken (optional),
  - and custom modules for document chunking and JSON loading.
"""

import os
import json
import logging
import time
import re
import asyncio
import hashlib
from functools import wraps

import numpy as np
import ollama
import faiss

embedding_cache = {}  # In-memory only

try:
    import tiktoken
except ImportError:
    tiktoken = None

from rank_bm25 import BM25Okapi
import networkx as nx

from utils.doc_chunker import semantically_chunk_documents
from utils.document_json_utils import load_documents_from_json

# Configs
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "deepseek-r1:32b-qwen-distill-q4_K_M")
SUMMARIZATION_MODEL = os.environ.get("SUMMARIZATION_MODEL", "qwen:32b")

DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", 0.5))
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", 0.5))

# Configure logging based on the verbose flag
# Read verbose flag from environment variable or set default to False
# export VERBOSE=True
VERBOSE = os.environ.get("VERBOSE", "False").lower() in ("true", "1", "yes")
logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,  
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# Retry Decorator (with exponential backoff)
def retry(max_attempts=3, initial_delay=1, backoff=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            delay = initial_delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.warning("Function %s failed with error: %s. Attempt %d/%d",
                                   func.__name__, e, attempts, max_attempts)
                    time.sleep(delay)
                    delay *= backoff
            raise Exception(f"Function {func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator

# async (wrappers for blocking API calls)
async def async_embedding(model: str, prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.embeddings(model=model, prompt=prompt))

async def async_generate(model: str, prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.generate(model=model, prompt=prompt))

# Token counting and summarization
def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken if available,
    otherwise use a simple word count.
    """
    if tiktoken:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning("tiktoken error: %s", e)
    return len(text.split())

@retry(max_attempts=3)
def summarize_chunk(chunk_content: str, max_tokens: int) -> str:
    """
    Summarize a chunk to fit within the token budget.
    Instead of naive truncation, we call the LLM to create a summary.
    """
    summary_prompt = (
        f"Summarize the following text in less than {max_tokens} tokens, "
        "keeping the most important details:\n\n"
        f"{chunk_content}"
    )
    response = ollama.generate(model=SUMMARIZATION_MODEL, prompt=summary_prompt)
    summary = response.get("response", "").strip()
    return summary if summary else chunk_content

# Candidate mergine (weighted averaging and deduplication)
def merge_candidates(dense_candidates: list, bm25_candidates: list) -> list:
    merged = {}
    # Process dense candidates (using similarity scores from FAISS).
    for candidate in dense_candidates:
        # candidate["similarity"] is assumed to be high when more similar.
        dense_score = DENSE_WEIGHT * candidate.get("similarity", 0)
        key = candidate["content"]
        merged[key] = {
            "content": candidate["content"],
            "metadata": candidate["metadata"],
            "dense_score": dense_score,
            "bm25_score": 0
        }
    # Process BM25 candidates.
    for candidate in bm25_candidates:
        bm25_score = BM25_WEIGHT * candidate.get("bm25_score", 0)
        key = candidate["content"]
        if key in merged:
            merged[key]["bm25_score"] = bm25_score
        else:
            merged[key] = {
                "content": candidate["content"],
                "metadata": candidate["metadata"],
                "dense_score": 0,
                "bm25_score": bm25_score
            }
    merged_list = []
    for cand in merged.values():
        combined = cand["dense_score"] + cand["bm25_score"]
        cand["combined_score"] = combined
        merged_list.append(cand)
    merged_list.sort(key=lambda x: x["combined_score"], reverse=True)
    # Deduplicate based on content hash.
    seen = set()
    deduped = []
    for cand in merged_list:
        h = hashlib.sha256(cand["content"].encode("utf-8")).hexdigest()
        if h not in seen:
            deduped.append(cand)
            seen.add(h)
    return deduped

def compute_hash(text: str) -> str:
    """Compute a SHA256 hash for a given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@retry(max_attempts=3)
def _get_embedding(text: str, model: str) -> list:
    """
    Retrieve an embedding for the text, caching it in-memory.
    """
    key = compute_hash(text)
    if key in embedding_cache:
        return embedding_cache[key]
    response = ollama.embeddings(model=model, prompt=text)
    embedding = response.get("embedding")
    if embedding:
        embedding_cache[key] = embedding
        return embedding
    else:
        logger.error("Embedding failed for text: %s", text[:30])
        raise ValueError("Embedding failed")

class LLMQueryEngine:
    """
    A robust LLM engine that implements a multi-stage retrieval-augmented generation pipeline:
      - Document loading and semantic chunking.
      - Embedding with in-memory caching.
      - In-memory FAISS index for dense retrieval.
      - BM25-based sparse retrieval.
      - A simple graph index for additional structure.
      - Hybrid retrieval and weighted candidate merging.
      - Dynamic context assembly with token budgeting and summarization.
      - Iterative context expansion.
      - Final answer generation using a Fusion-in-Decoder prompt.
    """

    def __init__(self,
                 json_file_path: str,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 embedding_model: str = EMBEDDING_MODEL,
                 generation_model: str = GENERATION_MODEL):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.generation_model = generation_model

        # Load and process documents.
        docs = load_documents_from_json(json_file_path)
        logger.info("Loaded %d documents from %s", len(docs), json_file_path)

        self.chunked_docs = self._chunk_documents(docs)
        logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

        # Build in-memory dense retrieval index using FAISS.
        self._build_faiss_index(self.chunked_docs)

        # Build BM25 index for sparse retrieval.
        self.bm25_corpus = [doc.page_content for doc in self.chunked_docs]
        self.bm25_tokens = [doc.page_content.lower().split() for doc in self.chunked_docs]
        self.bm25_index = BM25Okapi(self.bm25_tokens)
        logger.info("Built BM25 index for %d document chunks.", len(self.chunked_docs))

        # Build a simple graph index using networkx.
        self.graph = self.build_graph_index(self.chunked_docs)
        logger.info("Built graph index with %d nodes.", self.graph.number_of_nodes())

    def _chunk_documents(self, documents: list) -> list:
        """Chunk documents while preserving metadata."""
        return semantically_chunk_documents(documents, ollama_model=self.embedding_model)

    def _build_faiss_index(self, documents: list):
        """
        Compute embeddings for each document chunk, store them in memory,
        and build a FAISS index (in-memory only).
        """
        self.dense_embeddings = []
        self.dense_mapping = []  # Mapping from FAISS index to document chunks.
        for doc in documents:
            try:
                emb = _get_embedding(doc.page_content, self.embedding_model)
                self.dense_embeddings.append(emb)
                self.dense_mapping.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            except Exception as e:
                logger.exception("Error computing embedding for a document chunk: %s", str(e))
        if not self.dense_embeddings:
            raise ValueError("No embeddings computed.")
        # Convert to a NumPy array of type float32.
        dense_np = np.array(self.dense_embeddings).astype("float32")
        # Normalize for cosine similarity.
        norms = np.linalg.norm(dense_np, axis=1, keepdims=True)
        dense_np = dense_np / np.maximum(norms, 1e-10)
        d = dense_np.shape[1]
        # Create a FAISS index using inner product (cosine similarity).
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(dense_np)
        logger.info("Built FAISS index with %d vectors of dimension %d.", dense_np.shape[0], d)

    @retry(max_attempts=3)
    def retrieve_initial_chunks(self, query: str, max_chunks: int = 10) -> list:
        """
        Retrieve candidate chunks using the in-memory FAISS index.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Invalid query")
        try:
            q_emb = _get_embedding(query, self.embedding_model)
            q_emb_np = np.array(q_emb).astype("float32")
            # Normalize the query embedding.
            q_emb_np = q_emb_np / np.maximum(np.linalg.norm(q_emb_np), 1e-10)
            q_emb_np = q_emb_np.reshape(1, -1)
            D, I = self.faiss_index.search(q_emb_np, max_chunks)
            candidate_chunks = []
            for idx, score in zip(I[0], D[0]):
                # FAISS with inner product returns higher scores for better matches.
                candidate_chunks.append({
                    "content": self.dense_mapping[idx]["content"],
                    "similarity": score,
                    "metadata": self.dense_mapping[idx]["metadata"]
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

    def build_graph_index(self, chunked_docs: list) -> nx.Graph:
        """Build a simple graph index using shared section headers."""
        graph = nx.Graph()
        for i, doc in enumerate(chunked_docs):
            graph.add_node(i, content=doc.page_content, metadata=doc.metadata)
        sections = {}
        for i, doc in enumerate(chunked_docs):
            header = doc.metadata.get("section_header")
            if header:
                sections.setdefault(header, []).append(i)
        for header, indices in sections.items():
            for i in indices:
                for j in indices:
                    if i < j:
                        graph.add_edge(i, j)
        return graph

    def route_query(self, query: str, max_chunks: int = 15) -> list:
        """
        Logical routing that merges dense (FAISS) and BM25 results.
        """
        logger.info("Routing query to hybrid (FAISS dense + BM25) retrieval.")
        dense_candidates = self.retrieve_initial_chunks(query, max_chunks=max_chunks)
        bm25_candidates = self.retrieve_bm25_chunks(query, max_chunks=max_chunks)
        return merge_candidates(dense_candidates, bm25_candidates)

    def assemble_context(self, ranked_chunks: list, token_budget: int) -> list:
        """
        Assemble context chunks into the final prompt by grouping them
        by section header and using a round-robin strategy. When a chunk
        would exceed the token budget, it is summarized rather than truncated.
        """
        # Group candidates by section header.
        groups = {}
        for chunk in ranked_chunks:
            header = chunk["metadata"].get("section_header", "No Header")
            groups.setdefault(header, []).append(chunk)
        # Sort each group by combined score.
        for header in groups:
            groups[header].sort(key=lambda x: x["combined_score"], reverse=True)
        context = []
        total_tokens = 0
        # Round-robin merge from the groups.
        group_keys = list(groups.keys())
        idx = 0
        while total_tokens < token_budget and any(groups.values()):
            header = group_keys[idx % len(group_keys)]
            if groups[header]:
                candidate = groups[header].pop(0)
                candidate_tokens = count_tokens(candidate["content"])
                if total_tokens + candidate_tokens <= token_budget:
                    context.append(candidate)
                    total_tokens += candidate_tokens
                else:
                    remaining_tokens = token_budget - total_tokens
                    summarized = summarize_chunk(candidate["content"], max_tokens=remaining_tokens)
                    candidate["content"] = summarized
                    context.append(candidate)
                    total_tokens += count_tokens(summarized)
                    break
            idx += 1
        logger.info("Assembled context with ~%d tokens.", total_tokens)
        return context

    def iterative_context_expansion(self, query: str, current_context: list, token_budget: int, max_rounds: int = 3) -> list:
        """
        If the initial context does not fill enough of the token budget,
        iteratively expand it using additional candidate retrieval.
        """
        round_counter = 0
        while round_counter < max_rounds:
            current_token_count = sum(count_tokens(chunk["content"]) for chunk in current_context)
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
                response = ollama.generate(model=self.generation_model, prompt=expansion_prompt)
                keywords = response.get("response", "").strip()
                logger.info("Round %d expansion keywords: %s", round_counter + 1, keywords)
            except Exception as e:
                logger.exception("Error during iterative context expansion: %s", str(e))
                break
            if not keywords:
                break
            refined_query = f"{query} {keywords}"
            additional_candidates = self.route_query(refined_query, max_chunks=5)
            existing_contents = {chunk["content"] for chunk in current_context}
            for candidate in additional_candidates:
                if candidate["content"] not in existing_contents:
                    current_context.append(candidate)
                    existing_contents.add(candidate["content"])
                    current_token_count += count_tokens(candidate["content"])
                    if current_token_count >= token_budget:
                        break
            round_counter += 1
        return current_context
    
    @retry(max_attempts=3)
    def generate_response(self, query: str, context_chunks: list) -> str:
        """
        Generate the final answer using a Fusion-in-Decoder style prompt,
        aggregating all the retrieved evidence.
        """
        evidence_blocks = "\n\n".join(
            f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}"
            for chunk in context_chunks
        )
        full_prompt = (
            "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
            "Below are several context sections, each starting with a header. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
            "### Evidence Blocks:\n"
            f"{evidence_blocks}\n\n"
            f"Query: {query}\n"
            "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
            "with your answer listed as the value. For example, the last line of your output should be:\n"
            '{"answer": ["name1","name2","name3"]}'
        )
        full_prompt = (
            "You are a helpful assistant that answers technical questions by fusing evidence from multiple documents.\n"
            "Below are several context sections. Please read each section carefully and integrate the relevant information to answer the query.\n\n"
            "### Evidence Blocks:\n"
            f"{evidence_blocks}\n\n"
            "###"
            f"Query: {query}\n"
            "Do not abreviate answers."
            "Please provide an explanation first, and then on a new line, output a JSON array object that contains only one key 'answer' "
            "with your answer listed as the value. For example, the last line of your output should be:\n"
            """{"name":"John"}"""
            """{"age":30}"""
            """{"cars":["Ford", "BMW", "Fiat"]}"""
        )

        logger.info("Final prompt provided to LLM:\n%s", full_prompt)
        try:
            response = ollama.generate(model=self.generation_model, prompt=full_prompt)
            generated_text = response.get("response", "No response generated.").strip()

            print(generated_text)
            logger.info("Raw generated response: %s", generated_text)
            json_matches = re.findall(r'\{[^}]*\}', generated_text)
            for json_str in json_matches:
                try:
                    result_obj = json.loads(json_str)
                    if "answer" in result_obj:
                        answer = result_obj["answer"]
                        # if isinstance(answer, list):
                        #     return ", ".join(answer)
                        return answer
                except json.JSONDecodeError:
                    continue
            raise ValueError("No valid JSON with 'answer' key found.")
        except Exception as e:
            logger.exception("Error generating final response: %s", str(e))
            raise
            # return "Error generating response."

    def query(self, query: str, max_chunks: int = 15, token_budget: int = 1024) -> str:
        """
        The high-level query method that ties together:
          - Candidate retrieval,
          - Context assembly,
          - Iterative expansion, and
          - Final answer generation.
        """
        start_time = time.time()
        candidates = self.route_query(query, max_chunks=max_chunks)
        if not candidates:
            logger.warning("No candidate chunks retrieved for query: %s", query)
            return "No relevant documents found."
        context_chunks = self.assemble_context(candidates, token_budget)
        expanded_context = self.iterative_context_expansion(query, context_chunks, token_budget)
        answer = self.generate_response(query, expanded_context)
        elapsed_time = time.time() - start_time

        # Log instrumentation data.
        log_data = {
            "query": query,
            "elapsed_time": elapsed_time,
            "token_usage": sum(count_tokens(c['content']) for c in expanded_context)
        }
        logger.info(json.dumps(log_data))
        return answer

# Singleton instance
_engine_instance = None

def init_engine(doc_json_file_path: str, **kwargs) -> LLMQueryEngine:
    """
    Initialize the LLMQueryEngine (singleton).
    
    This function takes in json path that represents the list of document objects
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMQueryEngine(doc_json_file_path, **kwargs)
    return _engine_instance

def query_llm(query: str, max_chunks: int = 10, token_budget: int = 1024) -> str:
    """
    Public function to process a query.
    """
    global _engine_instance
    if _engine_instance is None:
        raise Exception("Engine not initialized. Please call init_engine(json_file_path) first.")
    return _engine_instance.query(query, max_chunks=max_chunks, token_budget=token_budget)

# For standalone testing
if __name__ == "__main__":
    json_file_path = "data/alexnet/doc_alexnet.json"
    engine = init_engine(
        json_file_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    queries = [
        ("""Name each layer of this neural network sequentially. Do not generalize internal layers and include modification and activation layers""", 1024),
    ]

    for idx, (q, budget) in enumerate(queries, start=1):
        start_time = time.time()
        answer = query_llm(q, token_budget=budget)
        elapsed = time.time() - start_time
        print(f"Example {idx}:\nAnswer: {answer}\nTime: {elapsed:.2f} seconds\n")