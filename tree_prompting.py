import json
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError


from utils.constants import Constants as C

from utils.query_rag_llm import LLM
from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import chunk_document
from utils.query_rag import DocumentIndexer
from utils.owl import *


def load_ontology_questions(json_path: str) -> dict:
    """
    Loads ontology questions from a JSON file.

    :param json_path: Path to the JSON file containing ontology questions.
    :return: Dictionary of ontology questions.
    """
    with open(json_path, 'r') as file:
        questions = json.load(file)
    return questions

# Load questions from the ontology JSON file
json_path = "rag/ontology_prompts.json"
questions_dict = load_ontology_questions(json_path)
QUESTIONS_DICT = questions_dict

if not QUESTIONS_DICT:
    raise ValueError(f"Failed to load ontology questions from {json_path}.")
    
class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology, conversation_tree, rag_query_engine):
        """
        Initializes the OntologyTreeQuestioner.

        :param ontology: Loaded ontology object.
        :param conversation_tree: Instance of the ConversationTree class.
        :param rag_query_engine: The LLM query engine for querying the language model.
        """
        self.ontology = ontology
        # Set the base class from which to start questioning
        self.base_class = ontology.ANNConfiguration
        self.conversation_tree = conversation_tree
        self.llm = rag_query_engine

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
                # Build the full prompt with context
                context_answers = self.get_context_answers(parent_id)
                if not context_answers:
                    # No context; use initial prompt
                    prompt = f"{get_initial_prompt()}\nQuestion:\n{question}"
                else:
                    # Display the ancestry history for debugging/logging
                    print("\nAncestry History:")
                    for i, (q, a) in enumerate(context_answers):
                        print(f"Step {i + 1}:")
                        print(f"  Question: {q}")
                        print(f"  Answer: {', '.join(a)}")

                    # Include context in the prompt
                    context_str = "\n".join(
                        [f"Step {i + 1}: Question: {q}\nAnswer: {', '.join(a)}" for i, (q, a) in enumerate(context_answers)]
                    )
                    prompt = f"{get_initial_prompt()}\nContext:\n{context_str}\nQuestion:\n{question}"

                # print(f"Prompt:\n{prompt}\n{'*' * 50}")

                # Query the LLM
                response = self.llm.query(prompt)
                response = str(response)
                validated_response = LLMResponse.parse_raw(response)

                # print(f"Response:\n{validated_response}\n","*"*50)

                return validated_response

            except (ValueError, ValidationError) as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        print("Max retries reached. Returning None.")
        return None

    # def get_context_answers(self, node_id):
    #     """
    #     Collects ancestor answers up to the given node to provide context.

    #     :param node_id: The ID of the current node.
    #     :return: A list of tuples containing questions and their corresponding answers.
    #     """
    #     answers = []
    #     current_id = node_id
    #     while current_id is not None:
    #         node = self.conversation_tree.nodes[current_id]
    #         if node['answer']:
    #             # Extract instance names from the answer
    #             instance_names = node['answer'].get('instance_names', [])
    #             if instance_names:
    #                 answers.append((node['question'], instance_names))
    #         current_id = node.get('parent_id')
    #     return list(reversed(answers))
    def get_context_answers(self, node_id):
        """
        Collects ancestor answers up to the given node to provide context.

        :param node_id: The ID of the current node.
        :return: A list of tuples containing questions and their corresponding answers.
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

        return list(reversed(answers))



    def ask_for_class(self, parent_id, cls, context=None, visited_classes=None):
        """
        Recursively asks about a class and its connected classes, updating the conversation tree.

        :param parent_id: ID of the parent node in the conversation tree.
        :param cls: The class to explore.
        :param context: A dictionary of ancestor class names and instance names.
        :param visited_classes: A set of classes that have already been visited to avoid cycles.
        """
        if cls is None:
            return
        if visited_classes is None:
            visited_classes = set()
        if cls in visited_classes:
            return
        visited_classes.add(cls)

        # Build the question with context
        question = get_ontology_question(cls.name, QUESTIONS_DICT)

        # # Add context to question if available
        # if context is not None:
        #     question = question.format(**context)
        # Add context to question if available
        if context is not None:
            try:
                question = question.format(**context)
            except KeyError as e:
                missing_key = str(e).strip("'")  # Extract the missing key
                print(f"Warning: Missing context key '{missing_key}' for class '{cls.name}'. Skipping...")
                return  # Skip this question if required context is missing

        # Ask the question to the LLM
        response_dict = self.ask_question(parent_id, question)
        if not response_dict or not response_dict.instance_names:
            # Class does not exist in the document; do not add to the conversation tree
            return

        # Transform `instance_names` into a dictionary with the class name as the key
        response_dict = {cls.name: instance_name for instance_name in response_dict.instance_names}

        # Add the class and instances to the conversation tree
        
        new_node_id = self.conversation_tree.add_child(parent_id, question, response_dict)

        # Update the context with the current class and its instances
        for class_name, instance_name in response_dict.items():
            new_context = context.copy() if context else {}
            new_context[cls.name] = instance_name

            # Get properties and connected classes
            direct_properties = get_class_properties(self.ontology, cls)
            connected_classes = get_property_class_range(direct_properties)

            # Recursively ask for each directly connected class
            for prop, classes in connected_classes.items():
                for cls_name in classes:

                    sub_cls = self.ontology[cls_name]
                    if sub_cls is None:
                        continue
                    self.ask_for_class(new_node_id, sub_cls, new_context, visited_classes)

    def start(self):
        """
        Starts the questioning process from the base class.
        """
        # Define the root question
        root_question = """
        Define the name of the architecture based on the given paper.
        Provide only the name.
        Example names for different papers could be 'GoogleLeNet', 'ResNet50', 'VGG16'. Only provide the name.
        If no names of that types are provided, respond with "ANNConfiguration".
        """

        root_class = self.ontology.Network

        root_question = get_ontology_question(root_class.name, QUESTIONS_DICT)

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
        direct_properties = get_class_properties(self.ontology, root_class)
        connected_classes = get_property_class_range(direct_properties)

        # Recursively ask for each directly connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                sub_cls = self.ontology[cls_name]
                if sub_cls is None:
                    continue
                self.ask_for_class(0, sub_cls, new_context)


        # direct_properties = get_class_properties(self.ontology,self.base_class)


        # connected_classes = get_property_class_range(direct_properties)

        # # Display the results
        # print("Direct properties and connected classes:")
        # for prop, classes in connected_classes.items():
        #     print(f"Property: {prop}, Connected Classes: {classes}")

        #  # Ask for each directly connected class
        # for prop, classes in connected_classes.items():
        #     for cls_name in classes:
        #         cls = self.ontology[cls_name]
        #         self.ask_for_class(1, cls)

class ConversationTree:
    """
    Manages and navigates a hierarchical conversation tree.
    """

    def __init__(self):
        self.tree = {"id": 0, "question": None, "answer": None, "children": [], "parent_id": None}
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}

    # def add_child(self, parent_id, question, answer=None):
    #     """
    #     Adds a child node to the conversation tree.

    #     :param parent_id: ID of the parent node.
    #     :param question: The question asked.
    #     :param answer: The answer received (optional).
    #     :return: The new node's ID.
    #     """
    #     self.node_id += 1
    #     child_node = {
    #         "id": self.node_id,
    #         "question": question,
    #         "answer": answer,
    #         "children": [],
    #         "parent_id": parent_id
    #     }
    #     self.nodes[parent_id]["children"].append(child_node)
    #     self.nodes[self.node_id] = child_node
    #     print(f"\nSaved question and answer in tree:\n{question}\n{answer}")
    #     return self.node_id
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



def get_ontology_question(entity: str, questions: dict) -> str:
    """
    Retrieves a question for a given entity.

    :param entity: The entity name (e.g., 'Dataset', 'batch_size').
    :param questions: Dictionary of ontology questions.
    :return: The associated question or a default message if not found.
    """
    entity_type = "ObjectPrompts"
    return questions.get(entity_type, {}).get(entity, f"No question found for '{entity}' in '{entity_type}'.")

# def get_initial_prompt() -> str:
#     initial_prompt = 
# """
#                     Work out your chain of thought.
#                     If you do not know the answer to a question, respond with "N/A." and nothing else.
#                     Question:
#                     """
#     return initial_prompt

def get_initial_prompt():
    return """
Please respond with a JSON object containing a single key, `instance_names`, 
which is a list of ontology class names relevant to the question.
Example:
{
    "instance_names": ["ResNet", "VGG16"]
}
    """


def get_second_prompt(question: str,response: str) -> str:
    second_prompt = f"""
                Given the question "{question}" and it's response "{response}", rephrase the response to follow these formatting rules:
                Single-item answers: If the question requires only one answer, respond with just that single answer.
                Single-value answers: If the question requires only one value, respond with just that single value.
                Listed answers: If the question requires multiple answers without order of priority, provide them as a list, with each answer as a single, complete item.
                Numbered list answers: If the question requires multiple answers in a specific sequence or hierarchy, provide them as a numbered list, with each number followed by a single, complete item.
                Use atomic answers for clarity, meaning each response should contain only one idea or concept per point. Ensure the format aligns with the nature of the question being asked.
                If multiple questions is asked, respond with a list in the order the questions were asked and nothing else. Do not label your answers.
                If the response says "N/A" or suggests that it does not know, reply with "N/A" and nothing else.

                Single-item answer example:
                Question: What is the capital of France?
                Answer: Paris

                Single-vlaue answer example:
                Question: How many states are in the United States?
                Answer: 50

                Listed answers example:
                Question: What are the primary colors in the American Flag?
                Answer: Red, White, Blue

                Numbered list example:
                Question: What are the steps in painting a wall?
                Answer: collect tools, 1 coat wall, 2 coat wall, 3 coat wall, wait to dry
                """
    return second_prompt

class LLMResponse(BaseModel):
    instance_names: List[str]  # A list of ontology class names expected from LLM



def main():
    pdf_path = "./data/papers/AlexNet.pdf"
    documents = load_pdf(pdf_path)
    chunked_docs = chunk_document(documents)
    embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
    llm_model = LLMModel(model_name="llama3.1:8b").get_llm()
    indexer = DocumentIndexer(embed_model, llm_model, chunked_docs)
    rag_query_engine = indexer.get_rag_query_engine()

    # Load ontology
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

    # Initialize the conversation tree
    tree = ConversationTree()

    # Initialize the OntologyTreeQuestioner
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        rag_query_engine=rag_query_engine
    )

    # Start the questioning process
    questioner.start()

    # Save the conversation tree to JSON
    tree.save_to_json("./rag/test_conversation_tree.json")








def print_instantiated_classes_and_properties(ontology: Ontology):
    """
    Prints all instantiated classes and their properties in the given ontology.

    Args:
        ontology (Ontology): The ontology to inspect.

    """
    print("Instantiated Classes and Properties:")
    for instance in ontology.individuals():
        print(f"Instance: {instance.name}")
        # Get the classes this instance belongs to
        classes = instance.is_a
        class_names = [cls.name for cls in classes if cls.name]
        print(f"  Classes: {', '.join(class_names) if class_names else 'None'}")
        
        # Get instantiated properties and their values
        properties = instance.get_properties()
        for prop in properties:
            values = instance.__getattr__(prop.name)
            if values:  # Only print properties that have values
                if isinstance(values, list):
                    values_str = ", ".join(str(v) for v in values)
                else:
                    values_str = str(values)
                print(f"  Property: {prop.name}, Values: {values_str}")
        print("-" * 40)

# Extra functions i wrote
def list_owl_classes(onto: Ontology):
    # List all classes
    print("Classes in the ontology:")
    for cls in onto.classes():
        print(f"- {cls.name}")

def list_owl_object_properties(onto: Ontology):
    # List all object properties
    print("\nObject Properties in the ontology:")
    for prop in onto.object_properties():
        print(f"- {prop.name}")

def list_owl_data_properties(onto: Ontology):
    # List all data properties
    print("\nData Properties in the ontology:")
    for prop in onto.data_properties():
        print(f"- {prop.name}")

def get_property_class_range(properties):
    """
    Retrieves the range (classes) of given properties.

    :param properties: List of properties whose ranges are to be found.
    :return: Dictionary mapping property names to lists of class names in their range.
    """
    connected_classes = {}
    for prop in properties:
        try:
            connected_classes[prop.name] = [
                cls.name for cls in prop.range if hasattr(cls, 'name')
            ]
        except AttributeError as e:
            print(f"Error processing property {prop.name}: {e}")
            connected_classes[prop.name] = []
    return connected_classes

if __name__ == "__main__":
    main()
