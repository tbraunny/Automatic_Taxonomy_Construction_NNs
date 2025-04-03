from typing import List, Optional, Dict, Any, Type
from pydantic import BaseModel


class PromptBuilder:
    """A utility class to construct LLM queries."""

    def __init__(self, general_header: str = None):
        self.general_header = general_header or (
            "You are an expert in neural network architectures with deep knowledge of various models, "
            "including CNNs, RNNs, Transformers, and more. Provide accurate, context-specific answers "
            "based on the provided information.\n\n"
        )
        self.examples: Dict[str, List[str]] = {}
        self.json_base_format = (
            'Return a JSON object with key "answer" containing the response. '
            "Use null for unavailable data unless specified otherwise."
        )

    def add_example(self, category: str, example: str) -> None:
        """Add an example for a specific prompt category."""
        if category not in self.examples:
            self.examples[category] = []
        self.examples[category].append(example)

    def build_prompt(
        self,
        task: str,
        query: str,
        category: str,
        extra_instructions: Optional[str] = None,
        include_examples: bool = True,
    ) -> str:
        """Construct a prompt with task, examples, and query."""
        parts = [self.general_header, f"Task: {task}\n"]

        if include_examples and category in self.examples:
            parts.append(
                "Formatting Examples:\n" + "\n".join(self.examples[category]) + "\n"
            )

        if extra_instructions:
            parts.append(f"Instructions: {extra_instructions}\n")

        parts.append(f"Query: {query}\n")
        parts.append(f"Format: {self.json_base_format}")

        return "\n".join(parts)

    def build_json_format_instructions(
        self, fields: Dict[str, str], example_output: Dict[str, Any]
    ) -> str:
        """Generate JSON format instructions from field descriptions and an example."""
        field_desc = "\n".join(f"- {k}: {v}" for k, v in fields.items())
        return (
            f"Return a JSON object with key 'answer' containing:\n{field_desc}\n\n"
            f"Example output:\n{example_output}"
        )
