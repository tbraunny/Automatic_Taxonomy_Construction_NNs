import json
from owlready2 import *
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from utils.constants import Constants as C
from utils.query_rag import RemoteDocumentIndexer
from utils.owl import *

class LLMResponse(BaseModel):
    instance_names: List[str]

class ConversationTree:
    """
    Manages and navigates a hierarchical conversation tree.
    """

    def __init__(self):
        self.tree = {
            "id": 0,
            "question": None,
            "answer": None,
            "children": [],
            "parent_id": None
        }
        self.node_id = 0
        self.nodes = {0: self.tree}

    def add_child(self, parent_id: Optional[int], question: str, answer: List[str]) -> int:
        """
        Adds a child node to the conversation tree.

        :param parent_id: ID of the parent node.
        :param question: The question asked.
        :param answer: The answer received.
        :return: The new node's ID.
        """
        self.node_id += 1

        child_node = {
            "id": self.node_id,
            "question": question,
            "answer": answer,
            "children": [],
            "parent_id": parent_id
        }
        if parent_id is not None:
            self.nodes[parent_id]["children"].append(child_node)
        self.nodes[self.node_id] = child_node
        return self.node_id

    def to_serializable(self, obj):
        """
        Converts the tree to a JSON-serializable structure.
        """
        if isinstance(obj, dict):
            return {key: self.to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(item) for item in obj]
        else:
            return obj

    def save_to_json(self, file_name: str):
        """
        Saves the conversation tree to a JSON file.
        """
        with open(file_name, "w") as f:
            json.dump(self.to_serializable(self.tree), f, indent=4)

class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology, conversation_tree, query_engine):
        self.ontology = ontology
        self.base_class = self.ontology.Network
        self.conversation_tree = conversation_tree
        self.llm = query_engine

    def ask_question(self, prompt: str, retries: int = 3) -> Optional[LLMResponse]:
        """
        Asks a question to the LLM and handles retries.

        :param prompt: The full prompt to send to the LLM.
        :param retries: Number of retries in case of failure.
        :return: Validated LLMResponse or None.
        """
        for attempt in range(retries):
            try:
                response_text = self.llm.query(prompt)
                validated_response = LLMResponse.parse_raw(response_text)
                return validated_response
            except (ValueError, ValidationError) as e:
                print(f"Validation error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
        print("Max retries reached. Returning None.")
        return None

    def get_context(self, node_id: int) -> str:
        """
        Collects ancestor questions and answers up to the given node.

        :param node_id: The ID of the current node.
        :return: A formatted string of context.
        """
        context = []
        current_id = node_id
        while current_id is not None:
            node = self.conversation_tree.nodes.get(current_id)
            if not node or not node['answer']:
                break
            question = node['question']
            answer = node['answer']
            context.append(f"Question: {question}\nAnswer: {', '.join(answer)}\n")
            current_id = node['parent_id']
        return "".join(reversed(context))

    def ask_for_class(self, parent_id: int, cls, visited_classes=None):
        """
        Recursively asks about a class and its connected classes.

        :param parent_id: ID of the parent node in the conversation tree.
        :param cls: The class to explore.
        :param visited_classes: Set of already visited classes.
        """
        if cls is None:
            return
        if visited_classes is None:
            visited_classes = set()
        if cls in visited_classes:
            return
        visited_classes.add(cls)

        # Build the question
        question_body = get_onto_prompt_with_examples(cls.name, PROMPTS_JSON)
        context_str = self.get_context(parent_id)
        full_prompt = f"{get_json_prompt()}\n"
        if context_str:
            full_prompt += f"Context:\n{context_str}\n"
        full_prompt += f"Question:\n{question_body}\nAnswer:"

        # Ask the question
        response = self.ask_question(full_prompt)
        if not response or not response.instance_names:
            return

        # Add to the conversation tree
        new_node_id = self.conversation_tree.add_child(parent_id, question_body, response.instance_names)

        # Get connected classes
        direct_properties = get_object_properties_for_class(self.ontology, cls)
        connected_classes = get_property_class_range(direct_properties)

        # Recursively ask for each connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                sub_cls = self.ontology.get(cls_name)
                if sub_cls is None:
                    continue
                self.ask_for_class(new_node_id, sub_cls, visited_classes)

    def start(self):

            # Initialize mapping from classes to node IDs
            class_to_node_id = {}

            # Process the root class first
            root_cls = self.base_class
            parent_cls_name = None

            # Build and format the question
            question_body = get_onto_prompt_with_examples(root_cls.name, PROMPTS_JSON)

            root_prompt = f"{get_json_prompt()}\n"
            root_prompt += f"Question:\n{question_body}\nAnswer:"

            # Prompt the LLM
            response = self.ask_question(root_prompt)
            if not response or not response.instance_names:
                return

            # Add to the conversation tree
            node_id = self.conversation_tree.add_child(0, question_body, response.instance_names)
            class_to_node_id[root_cls] = node_id  # Map the root class to its node ID

            # Now process subclasses
            for cls in iterate_subclasses(root_cls, onto=self.ontology):
                if cls == root_cls:
                    continue  # Skip the root class as it's already processed

                        # Determine the parent class
                parent_classes = cls.is_a
                parent_class = parent_classes[0] if parent_classes else None

                # Get parent node ID and instance names
                parent_node_id = class_to_node_id.get(parent_class, 0)

                # Format the question with the parent class name
                question_body = get_onto_prompt_with_examples(cls.name, PROMPTS_JSON)
                # question_body = question_body.replace("{parent_class}", parent_class_name)

                # Get the parent node ID
                parent_node_id = class_to_node_id.get(parent_class, 0)

                # Build the prompt
                context_str = self.get_context(parent_node_id)
                full_prompt = f"{get_json_prompt()}\n"
                if context_str:
                    full_prompt += f"Context:\n{context_str}\n"
                full_prompt += f"Question:\n{question_body}\nAnswer:"

                # Prompt the LLM
                response = self.ask_question(full_prompt)
                if not response or not response.instance_names:
                    continue

                # Add to the conversation tree
                node_id = self.conversation_tree.add_child(parent_node_id, question_body, response.instance_names)
                class_to_node_id[cls] = node_id


def main():
    query_engine = RemoteDocumentIndexer('100.86.58.97', 5000).get_rag_query_engine()
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
    prompts_json = load_ontology_questions("rag/ontology_prompts.json")    
    global PROMPTS_JSON
    PROMPTS_JSON = prompts_json

    tree = ConversationTree()
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        query_engine=query_engine
    )
    questioner.start()
    tree.save_to_json("./rag/conversation_tree.json")

""" Helper functions """

def load_ontology_questions(json_path: str) -> dict:
    """
    Loads ontology questions from a JSON file.
    """
    with open(json_path, 'r') as file:
        questions = json.load(file)
    return questions

def get_json_prompt() -> str:
    return """
Please respond with a JSON object containing a single key, `instance_names`, 
which is a list of ontology class names relevant to the question.
Example:
{
    "instance_names": ["ResNet", "VGG16"]
}
    """

def get_onto_prompt_with_examples(entity: str, ontology_data: List[dict]) -> str:
    """
    Combines the prompt and its few-shot examples into a single formatted string.
    """
    for entry in ontology_data:
        if entry.get("class") == entity:
            prompt = entry.get("prompt", f"No prompt found for '{entity}'.")
            examples = entry.get("few_shot_examples", [])
            combined = f"{prompt}\n\n"
            for idx, example in enumerate(examples, 1):
                combined += f"Example {idx}:\n"
                combined += f"  Question: {example['input']}\n"
                combined += f"  Answer: {example['output']}\n\n"
            return combined.strip()
    return f"No entry found for '{entity}' in the ontology."


if __name__ == "__main__":
    main()
