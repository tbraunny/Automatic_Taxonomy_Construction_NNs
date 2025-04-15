import os
import time
import numpy as np
import faiss
import ollama
import asyncio
from typing import Type, Any, TypeVar
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from rank_bm25 import BM25Okapi

from utils.doc_chunker import semantically_chunk_documents
from utils.document_json_utils import load_documents_from_json
from utils.logger_util import get_logger

T = TypeVar("T", bound=BaseModel)

from utils.pydantic_models import *

# In-memory embedding cache
embedding_cache = {}

# Configs
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "bge-m3")
# GENERATION_MODEL = os.environ.get(
#     "GENERATION_MODEL", "neuroexpert:latest"
# )
# GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwq:32b-q4_K_M")
# GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "qwen2.5:32b")
# GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "deepseek-r1:32b-qwen-distill-q4_K_M")
# GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "command-r")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "deepseek-r1:1.5b")

logger = get_logger("llm_service")

DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", 0.5))
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", 0.5))

# async (wrappers for blocking API calls)
async def async_embedding(model: str, prompt: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ollama.embeddings(model=model, prompt=prompt))

def compute_hash(text: str) -> str:
    """Compute a SHA256 hash for a given text."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embedding(text: str, model: str) -> list:
    """Retrieve an embedding for the text, caching it in-memory."""
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


from langchain_core.callbacks.base import BaseCallbackHandler

class DebugCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, prompts, **kwargs):
        print("LLM Start:", prompts)
        print("\n\n\n")

    def on_llm_end(self, output, **kwargs):
        print("LLM Debug Output:", output)
        logger.debug(f"LLM Debug Output: {output}")
        print("\n\n\n")

    def on_text(self, text, **kwargs):
        print("LLM Text:", text)
        print("\n\n\n")
        logger.debug(f"LLM Text: {text}")
        print("\n\n\n")

    def on_llm_error(self, error, **kwargs):
        print("LLM Error:", error)
        print("\n\n\n")

    def on_tool_start(self, serialized, **kwargs):
        print("Tool Start:", serialized)
        print("\n\n\n")

    def on_tool_end(self, output, **kwargs):
        print("Tool End:", output)
        print("\n\n\n")

    def on_tool_error(self, error, **kwargs):
        print("Tool Error:", error)
        print("\n\n\n")

    def on_rety(self, retry_state, **kwargs):
        print("Retry:", retry_state)
        print("\n\n\n")

class LLMQueryEngine:

    def __init__(
        self,
        json_file_path: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
        generation_model: str = GENERATION_MODEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.logger = logger

        self.llm = ChatOllama(
            # base_url="http://localhost:11434",
            model=self.generation_model,
            temperature=0.2,
            seed=42,
            num_ctx=10000,
            # verbose=True,
            # callbacks=[DebugCallbackHandler()], # Uncomment for debugging ollama server
        )

        # Load and chunk documents
        docs = load_documents_from_json(json_file_path)
        self.logger.info("Loaded %d documents from %s", len(docs), json_file_path)
        self.chunked_docs = semantically_chunk_documents(
            docs, ollama_model=self.embedding_model
        )
        self.logger.info("Chunked documents into %d chunks", len(self.chunked_docs))

        # Build in-memory FAISS index
        self._build_faiss_index(self.chunked_docs)

        # Build BM25 index
        self.bm25_corpus = [doc.page_content for doc in self.chunked_docs]
        self.bm25_tokens = [
            doc.page_content.lower().split() for doc in self.chunked_docs
        ]
        self.bm25_index = BM25Okapi(self.bm25_tokens)
        self.logger.info(
            "Built BM25 index for %d document chunks.", len(self.chunked_docs)
        )

    def _build_faiss_index(self, documents: list):
        """Build an in-memory FAISS index for dense retrieval."""
        self.dense_embeddings = []
        self.dense_mapping = []
        for doc in documents:
            try:
                emb = get_embedding(doc.page_content, self.embedding_model)
                self.dense_embeddings.append(emb)
                self.dense_mapping.append(
                    {"content": doc.page_content, "metadata": doc.metadata}
                )
            except Exception as e:
                self.logger.exception("Error computing embedding: %s", str(e))
        if not self.dense_embeddings:
            raise ValueError("No embeddings computed.")
        # Convert to a NumPy array of type float32.
        dense_np = np.array(self.dense_embeddings).astype("float32")
        # Normalize for cosine similarity.
        norms = np.linalg.norm(dense_np, axis=1, keepdims=True)
        dense_np = dense_np / np.maximum(norms, 1e-10)
        d = dense_np.shape[1]
        # Create a FAISS index using cosine similarity.
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(dense_np)
        self.logger.info(
            "Built FAISS index with %d vectors of dimension %d.", dense_np.shape[0], d
        )

    def retrieve_dense_chunks(self, query: str, max_chunks: int = 10) -> list:
        """Retrieve relevant chunks using FAISS."""
        if not query or not isinstance(query, str):
            raise ValueError("Invalid query")
        try:
            q_emb = get_embedding(query, self.embedding_model)
            q_emb_np = np.array(q_emb).astype("float32")
            q_emb_np = q_emb_np / np.maximum(np.linalg.norm(q_emb_np), 1e-10)
            q_emb_np = q_emb_np.reshape(1, -1)
            D, I = self.faiss_index.search(q_emb_np, max_chunks)
            candidate_chunks = []
            for idx, score in zip(I[0], D[0]):
                candidate_chunks.append(
                    {
                        "content": self.dense_mapping[idx]["content"],
                        "similarity": score * DENSE_WEIGHT,
                        "metadata": self.dense_mapping[idx]["metadata"],
                    }
                )
            return candidate_chunks
        except Exception as e:
            self.logger.exception("Error retrieving dense chunks: %s", str(e))
            return []

    def retrieve_bm25_chunks(self, query: str, max_chunks: int = 10) -> list:
        """Retrieve candidate chunks using BM25."""
        tokens = query.lower().split()
        scores = self.bm25_index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :max_chunks
        ]
        candidates = []
        for i in top_indices:
            doc = self.chunked_docs[i]
            candidates.append(
                {
                    "content": doc.page_content,
                    "similarity": scores[i] * BM25_WEIGHT,
                    "metadata": doc.metadata,
                }
            )
        return candidates

    def merge_candidates(
        self, dense_candidates: List, bm25_candidates: List
    ) -> List:
        """Merge dense and BM25 candidates with weighted scoring."""
        merged = {}
        for candidate in dense_candidates:
            key = compute_hash(candidate["content"])
            merged[key] = {
                "content": candidate["content"],
                "metadata": candidate["metadata"],
                "score": candidate["similarity"],
            }
        for candidate in bm25_candidates:
            key = compute_hash(candidate["content"])
            if key in merged:
                # Add BM25 score to existing dense score
                merged[key]["score"] += candidate["similarity"]
                self.logger.debug(
                    "Duplicate chunk found and merged: %s", candidate["content"][:30]
                )
            else:
                merged[key] = {
                    "content": candidate["content"],
                    "metadata": candidate["metadata"],
                    "score": candidate["similarity"],
                }
        merged_list = list(merged.values())
        return sorted(merged_list, key=lambda x: x["score"], reverse=True)

    def retrieve_chunks(self, query: str, max_chunks: int = 10) -> list:
        """Retrieve and merge chunks from FAISS and BM25."""
        dense_candidates = self.retrieve_dense_chunks(query, max_chunks)
        bm25_candidates = self.retrieve_bm25_chunks(query, max_chunks)
        if not dense_candidates and not bm25_candidates:
            self.logger.warning("No candidates found for query: %s", query)
            return []
        if not dense_candidates:
            self.logger.info("No dense candidates found, using BM25 only.")
            return bm25_candidates
        if not bm25_candidates:
            self.logger.info("No BM25 candidates found, using dense only.")
            return dense_candidates
        self.logger.info(
            f"Merging {len(dense_candidates)} dense candidates and {len(bm25_candidates)} BM25 candidates.",
        )
        return self.merge_candidates(dense_candidates, bm25_candidates)

    def assemble_context(
        self, chunks: list, token_budget: int
    ) -> str:  # NOTE: Fixed duplicate chunk issue
        """Assemble context within token budget."""
        if not chunks:
            self.logger.error("No chunks to assemble context from.",exc_info=True)
            raise ValueError("No chunks to assemble context from.")
        if token_budget <= 1500:
            new_token_budget = 3000
            self.logger.warning(f"Strangly small token budget of {token_budget}, changing to {new_token_budget} tokens.")
            token_budget = new_token_budget

        context = ""
        skipped = 0
        total_words = 0
        seen_content = set()  # Track content hashes to avoid duplicates
        for chunk in chunks:
            try:
                if not chunk["content"].strip():
                    self.logger.debug("Skipping empty chunk.")
                    continue
                content_hash = compute_hash(chunk["content"])
                if content_hash in seen_content:
                    self.logger.debug("Skipping duplicate chunk: %s out of total %s", chunk["content"][:30], skipped)
                    skipped += 1
                    continue
                chunk_words = len(chunk["content"].split())
                if total_words + chunk_words <= token_budget:
                    
                    context += f"### {chunk['metadata'].get('section_header', 'No Header')}\n{chunk['content']}\n\n"
                    total_words += chunk_words
                    seen_content.add(content_hash)
                else:
                    break
            except Exception as e:
                self.logger.exception("Error processing chunk: %s", str(e))
                continue
        self.logger.info(
            "Assembled context with ~%d words from %d unique chunks.",
            total_words,
            len(seen_content),
        )
        return context.strip()
    

    def query_structured(
        self,
        query: str,
        cls_schema: Type[BaseModel],
        token_budget: int,
        max_chunks: int,
    ) -> T:
        """Query the LLM with structured output using ChatOllama."""
        start_time = time.time()
        candidates = self.retrieve_chunks(query, max_chunks)
        self.logger.info(
            f"Retrieved {len(candidates)} candidate chunks for query."
        )
        if not candidates:
            self.logger.error(f"No candidate chunks retrieved for query: {query}", exc_info=True)
            raise ValueError(f"No candidate chunks retrieved for query: {query}")

        context = self.assemble_context(candidates, token_budget)

        if not context:
            self.logger.error("No context assembled for query: %s", query, exc_info=True)
            raise ValueError("No context assembled for query: %s", query)
        self.logger.info(
            f"Assembled context for query."
        )
        prompt = (
            f"Goal: Answer the query based on the provided context.\n\n"
            f"Query: {query}\n\n"
            f"Context:\n```\n{context}\n```\n\n"
            f"Response:\n"
        ).strip()
        self.logger.info(
            "Generated prompt:\n%s",
            prompt,
        )

        structured_llm = self.llm.with_structured_output(cls_schema)
        try:
            response = structured_llm.invoke(prompt)

            self.logger.info("Raw LLM response: %s", response)
            if not response:
                self.logger.error("LLM returned None for query: %s", query)

                raise ValueError("LLM returned None")

            # Ensure response is a valid Pydantic model instance
            if not isinstance(response, cls_schema):
                self.logger.error(
                    "Response type mismatch. Expected %s, got %s",
                    cls_schema,
                    type(response),
                )

                raise ValueError(
                    f"Response type mismatch. Expected {cls_schema}, got {type(response)}"
                )

            elapsed_time = time.time() - start_time
            self.logger.info(
                "Time Elapsed for Query: %.2f seconds",
                elapsed_time,
            )
            return response
        except Exception as e:
            self.logger.exception("Error generating structured response: %s", str(e))
            raise


# Engine instance management
_engine_instances = {}


def init_engine(model_name: str, doc_json_file_path: str, **kwargs) -> LLMQueryEngine:
    """Initialize and cache an LLMQueryEngine instance."""
    if model_name not in _engine_instances:
        _engine_instances[model_name] = LLMQueryEngine(doc_json_file_path, **kwargs)
    return _engine_instances[model_name]


def query_llm(
    model_name: str,
    query: str,
    cls_schema: Type[BaseModel],
    max_chunks: int = 10,
    token_budget: int = 3000,
) -> Any:
    """High-level query function with structured output."""
    if model_name not in _engine_instances:
        raise Exception(
            f"Engine for model '{model_name}' not initialized. Call init_engine first."
        )
    engine = _engine_instances[model_name]
    return engine.query_structured(query, cls_schema, token_budget, max_chunks)


# Example Pydantic models for testing
class Layer(BaseModel):
    type: str
    size: int


class NeuralNetworkDetails(BaseModel):
    layers: list[Layer]


# Standalone testing
if __name__ == "__main__":
    json_file_path = "data/alexnet/doc_alexnet.json"
    engine = init_engine(
        model_name=GENERATION_MODEL,
        doc_json_file_path=json_file_path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    queries = [
        # ("""Name each layer of this neural network sequentially. Do not generalize internal layers and include modification and activation layers""", 1024),
        (
            """Name each layer of this neural network sequentially. Do not generalize internal layers and include modification and activation layers. Do not generalize internal layers and include modification and activation layers. Also, specify the size of each layer (number of neurons or filters).""",
            7000,
        ),
    ]

    json_format_instructions = (
        "Return a JSON object with a key 'layers'. The value should be a list of objects, "
        "where each object represents one layer in the neural network. Each object must have exactly two keys: "
        "'type' (a string, e.g., 'Convolution', 'ReLU', 'Pooling', 'FullyConnected', etc.) and "
        "'size' (an integer representing the number of neurons/filters in that layer). "
        "Ensure the response is valid JSON that matches this structure.\n"
        "Example:\n"
        '{"layers": [{"type": "Convolution", "size": 64}, {"type": "ReLU", "size": 64}]}'
    )

    for idx, (q, budget) in enumerate(queries, start=1):
        start_time = time.time()
        answer = query_llm(
            GENERATION_MODEL,
            q,
            json_format_instructions,
            NeuralNetworkDetails,
            token_budget=budget,
        )
        print("\n\nStructured Layer Details:\n")
        for i, layer in enumerate(answer.layers, start=1):
            layer = layer.model_dump()
            print(f"Layer {i}: Type = {layer['type']}, Size = {layer['size']}")

        elapsed = time.time() - start_time
        print(f"Example {idx}:\nAnswer: {answer}\nTime: {elapsed:.2f} seconds\n\n\n")
