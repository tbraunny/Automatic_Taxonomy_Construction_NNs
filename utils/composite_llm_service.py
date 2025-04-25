from utils.llm_service import LLMQueryEngine, init_engine
import time
from pydantic import BaseModel
from typing import Type, List, Optional, TypeVar
T = TypeVar("T", bound=BaseModel)

class CompositeLLMQueryEngine:
    def __init__(self, engines: list[LLMQueryEngine], logger):
        self.engines = engines
        self.logger = logger

    def retrieve_chunks(self, query: str, max_chunks: int = 10) -> list:
        all_candidates = []
        for engine in self.engines:
            candidates = engine.retrieve_chunks(query, max_chunks)
            all_candidates.extend(candidates)

        merged_candidates = sorted(all_candidates, key=lambda x: x["score"], reverse=True)
        return merged_candidates[:max_chunks]

    def assemble_context(self, chunks: list, token_budget: int) -> str:
        return self.engines[0].assemble_context(chunks, token_budget)

    def query_structured(
        self,
        query: str,
        cls_schema: Type[BaseModel],
        token_budget: int = 3000,
        max_chunks: int = 10,
    ) -> T:
        """Query across all engines"""
        start_time = time.time()
        candidates = self.retrieve_chunks(query, max_chunks)
        if not candidates:
            raise ValueError(f"No candidate chunks retrieved for query: {query}")

        context = self.assemble_context(candidates, token_budget)

        prompt = (
            f"Goal: Answer the query based on the provided context.\n\n"
            f"Query: {query}\n\n"
            f"Context:\n```\n{context}\n```\n\n"
            f"Response:\n"
        ).strip()

        # Assume all engines use same client type
        structured_llm = self.engines[0].llm_client.llm.with_structured_output(cls_schema)
        response = structured_llm.invoke(prompt)

        if not isinstance(response, cls_schema):
            raise ValueError(f"Response type mismatch. Expected {cls_schema}, got {type(response)}")
        
        elapsed_time = time.time() - start_time
        self.logger.info("Time Elapsed for Composite Query: %.2f seconds", elapsed_time)
        return response

if __name__ == "__main__":

    class SourceReference(BaseModel):
        source_title: Optional[str] = None
        section_header: Optional[str] = None
        metadata: Optional[dict[str, str]] = None 


    class UserAnswer(BaseModel):
        answer_text: str
        references: Optional[List[SourceReference]] = None
        notes: Optional[str] = None

    list_json_doc_paths = {"alexnet":"data/alexnet/alexnet_doc.json", "resnet":"data/resnet/resnet_doc.json", "vgg":"data/vgg16/vgg16_doc.json"}
    query = "What is the loss function used in alexnet."


    init_engines = []
    for ann_name, json_doc_path in list_json_doc_paths.items():
        init_engines.append(init_engine(ann_name, json_doc_path))

    composite_engine = CompositeLLMQueryEngine(
        engines=init_engines,
        logger = init_engines[0].logger
    )
    result = composite_engine.query_structured(
        query=query,
        cls_schema=UserAnswer,
        token_budget=7000,
        max_chunks=20,
    )