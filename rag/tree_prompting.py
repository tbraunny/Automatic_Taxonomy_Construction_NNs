import json
from owlready2 import *
from typing import List, Union
from pydantic import BaseModel, ValidationError
from utils.constants import Constants as C
from utils.query_rag import RemoteDocumentIndexer
from utils.owl import *

'''
scispacy nlp by size
    en_core_sci_sm
    en_core_sci_md
    en_core_sci_lg 
'''
class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology, conversation_tree, query_engine):
        """
        Initializes the OntologyTreeQuestioner.

        :param ontology: Loaded ontology object.
        :param conversation_tree: Instance of the ConversationTree class.
        :param rag_query_engine: The LLM query engine for querying the language model.
        """
        self.ontology = ontology
        self.base_class = get_base_class(ontology)
        self.conversation_tree = conversation_tree
        self.llm = query_engine

    def ask_question(self, parent_id, question, retries=3):
        """
        Asks a question to the LLM, incorporating context from ancestor nodes, and handles retries.

        :param parent_id: ID of the parent node in the conversation tree.
        :param question: The question to ask the LLM.
        :param retries: Number of times to retry in case of failure. Default is 3.
        :return: The validated response from the LLM, or None if max retries reached.
        """
        for attempt in range(retries):
            try:
                # Retrieve context
                context_answers = self.get_context(parent_id)
                if not context_answers:
                    # No context; use initial prompt
                    prompt = f"{get_json_prompt()}\nQuestion:\n{question}"
                else:
                    # Format context into a string
                    context_str = self.get_context(parent_id)
                    # Combine context and the main question
                    prompt = f"{get_json_prompt()}\nContext:\n{context_str}\nQuestion:\n{question}"

                # Query the LLM
                response = self.llm.query(prompt)
                response = str(response)
                validated_response = LLMResponse.parse_raw(response)
                return validated_response

            except (ValueError, ValidationError) as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        print("Max retries reached. Returning None.")
        return None

    
    def get_context(self, node_id):
        """
        Collects ancestor answers up to the given node and formats them as a string to provide context.

        :param node_id: The ID of the current node.
        :return: A formatted string of questions and their corresponding answers, or an empty string if no context exists.
        """
        answers = []
        current_id = node_id
        while current_id is not None:
            print(f"Retrieving context for node ID: {current_id}")  # Debugging log
            node = self.conversation_tree.nodes.get(current_id)
            if not node:
                print(f"Node with ID {current_id} not found. Stopping traversal.")
                break

            if node['answer']:
                instance_names = node['answer'].get('instance_names', [])
                if instance_names:
                    answers.append((node['question'], instance_names))

            current_id = node.get('parent_id')  # Move to parent node

        if not answers:
            print("No context found during traversal.")  # Additional debug log
            return ""

        # Format the context as a string
        context_str = "\n".join(
            [f"Step {i + 1}: Question: {q}\nAnswer: {', '.join(a)}"
            for i, (q, a) in enumerate(reversed(answers))]
        )

        return context_str


    def ask_for_class(self, parent_id, cls, visited_classes=None):
        """
        Recursively asks about a class and its connected classes, updating the conversation tree.

        :param parent_id: ID of the parent node in the conversation tree.
        :param cls: The class to explore.
        :param visited_classes: A set of classes that have already been visited to avoid cycles.
        """
        if cls is None:
            return
        if visited_classes is None:
            visited_classes = set()
        if cls in visited_classes:
            return
        visited_classes.add(cls)

        # Fetch context dynamically based on the parent node
        context_str = self.get_context(parent_id)

        # Build the question
        question = get_onto_prompt_with_examples(cls.name, PROMPTS_JSON)
        if context_str:
            question = f"Context:\n{context_str}\n{question}"

        # Ask the question
        response_dict = self.ask_question(parent_id, question)
        if not response_dict or not response_dict.instance_names:
            # If no instances are returned, skip further exploration
            return

        # Transform `instance_names` into a dictionary with the class name as the key
        response_dict = {cls.name: instance_name for instance_name in response_dict.instance_names}

        # Add the class and its instances to the conversation tree
        new_node_id = self.conversation_tree.add_child(parent_id, question, response_dict)

        # Get properties and connected classes
        direct_properties = get_object_properties_for_class(self.ontology, cls)
        connected_classes = get_property_class_range(direct_properties)

        # Recursively ask for each directly connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                sub_cls = self.ontology.get(cls_name)
                if sub_cls is None:
                    continue
                self.ask_for_class(new_node_id, sub_cls, visited_classes)


    def start(self):
        """
        Starts the questioning process from the base class.
        """

        root_class = self.ontology.Network

        print(root_class, root_class.name)

        root_question = get_onto_prompt_with_examples(root_class.name, PROMPTS_JSON)


        print(root_question)
        # Ask the root question
        response_dict = self.ask_question(0, root_question)

        if not response_dict or not response_dict.instance_names:
            return

        # Populate the root node with the question and answer
        self.conversation_tree.nodes[0]['question'] = root_question
        self.conversation_tree.nodes[0]['answer'] = {'instance_names': response_dict.instance_names}

        # Update the context with the root class and its instances
        new_context = {root_class.name: response_dict.instance_names[0]}

        # Get properties and connected classes
        direct_properties = get_object_properties_for_class(root_class, self.ontology)
        connected_classes = get_property_class_range(direct_properties)

        # Recursively ask for each directly connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                sub_cls = self.ontology[cls_name]
                if sub_cls is None:
                    continue
                self.ask_for_class(0, sub_cls, new_context)

class ConversationTree:
    """
    Manages and navigates a hierarchical conversation tree.
    """

    def __init__(self):
        self.tree = {"id": 0, "question": None, "answer": None, "children": [], "parent_id": None}
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}

    def add_child(self, parent_id, question, answer=None):
        """
        Adds a child node to the conversation tree.

        :param parent_id: ID of the parent node.
        :param question: The question asked.
        :param answer: The answer received (optional).
        :return: The new node's ID.
        """
        self.node_id += 1

        child_node = {
            "id": self.node_id,
            "question": question,
            "answer": answer,
            "children": [],
            "parent_id": parent_id  # Link this node to its parent
        }
        self.nodes[parent_id]["children"].append(child_node)  # Add child to parent's list
        self.nodes[self.node_id] = child_node  # Store the new node in the tree
        return self.node_id


    def to_serializable(self, obj):
        """
        Converts non-serializable objects to a JSON-compatible structure.

        :param obj: The object to convert.
        :return: A JSON-serializable structure.
        """
        from requests.models import Response
        if isinstance(obj, dict):
            return {key: self.to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self.to_serializable(vars(obj))
        elif isinstance(obj, Response):  # Handle the `Response` object?????
            return {"response_text": str(obj)}
        else:
            return obj

    def save_to_json(self, file_name):
        """
        Saves the conversation tree to a JSON file.

        :param file_name: Path to the file where the tree will be saved.
        """
        with open(file_name, "w") as f:
            json.dump(self.to_serializable(self.tree), f, indent=4)



class LLMResponse(BaseModel):
    instance_names: List[str]  # A list of ontology class names expected from LLM



def main():
    query_engine = RemoteDocumentIndexer('100.105.5.55',5000).get_rag_query_engine()
    question = query_engine.query("hellow llama")
    print(question)
    return
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
    prompts_json = load_ontology_questions("rag/ontology_prompts.json")    
    global PROMPTS_JSON
    PROMPTS_JSON = prompts_json


    # prompt = get_onto_prompt_with_examples(entity_to_query, PROMPTS_JSON)

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

    :param json_path: Path to the JSON file containing ontology questions.
    :return: Dictionary of ontology questions.
    """
    with open(json_path, 'r') as file:
        questions = json.load(file)
    return questions


def get_chain_of_thought_prompt():
    return"""
Work out your chain of thought.
If you do not know the answer to a question, respond with "N/A" and nothing else.
"""

def get_json_prompt():
    return """
Please respond with a JSON object containing a single key, `instance_names`, 
which is a list of ontology class names relevant to the question.
Example:
{
    "instance_names": ["ResNet", "VGG16"]
}
    """


def get_onto_prompt(entity: str, ontology_data: List[dict]) -> str:
    """
    Retrieves a question (prompt) for a given entity from ontology data, ignoring object properties.

    :param entity: The entity name (e.g., 'Network', 'Layer').
    :param ontology_data: A list of dictionaries representing the ontology JSON structure.
    :return: The associated question (prompt) or a default message if not found.
    """
    for entry in ontology_data:
        # Ignore entries that contain 'object_property'
        if "object_property" in entry:
            continue
        if entry.get("class") == entity:
            return entry.get("prompt", f"No prompt found for '{entity}'.")
    return f"No entry found for '{entity}' in the ontology."


def get_onto_prompt_with_examples(entity: str, ontology_data: List[dict]) -> str:
    """
    Combines the prompt and its few-shot examples into a single formatted string.

    :param entity: The entity name (e.g., 'Network', 'LossFunction').
    :param ontology_data: A list of dictionaries representing the ontology JSON structure.
    :return: A well-structured string combining the prompt and examples, or a message if not found.
    """
    for entry in ontology_data:
        if entry.get("class") == entity:
            # Get the prompt
            prompt = entry.get("prompt", f"No prompt found for '{entity}'.")
            # Get the few-shot examples
            examples = entry.get("few_shot_examples", [])
            
            if not examples:
                return f"Question: {prompt}\nAnswer:'."

            # Start the combined string
            combined = f"Question: {prompt}\n\n"

            # Format each example in a clear and consistent manner
            for idx, example in enumerate(examples, 1):
                combined += f"Example {idx}:\n"
                combined += f"  Question: {example['input']}\n"
                combined += f"  Answer: {example['output']}\n\n"

            combined += f"Question:{prompt}\nAnswer:"
            return combined.strip()
    return f"No entry found for '{entity}' in the ontology."

if __name__ == "__main__":
    main()
