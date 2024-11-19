import json
from owlready2 import *
from typing import List, Dict

from utils.constants import Constants as C

from utils.query_llm import LLM
from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import chunk_document
from utils.document_indexer import DocumentIndexer

from utils.owl import *

QUESTIONS_DICT = None
    
class OntologyTreeQuestioner:
    """
    A class to generate questions based on ontology classes and their properties,
    integrate with a conversation tree, and recursively ask questions for object properties.
    """

    def __init__(self, ontology, conversation_tree, rag_query_engine):
        """
        Initialize with ontology, base class, and conversation tree.
        
        :param ontology: Loaded ontology object.
        :param base_class: Base class to start questioning.
        :param conversation_tree: Instance of the ConversationTree class.
        """
        self.ontology = ontology
        # Get base class
        self.base_class = ontology.ANNConfiguration
        print(f"Base class: {self.base_class.name}")

        self.conversation_tree = conversation_tree
        self.llm=rag_query_engine

    def ask_question(self, parent_id, question):
        """
        Asks question to LLM
        
        :param parent_id: ID of the parent node in the conversation tree.
        :param question: The question to ask.
        :return: Answer as a string.
        """

        query = "".join([get_initial_prompt(),question])
        response = self.llm.query(query)
        final_response = self.llm.query(get_second_prompt(question,response)) 

        print("Question:")
        print(question)
        print("-" * 50)
        print("First Answer:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print("Second Answer:")
        print("-" * 50)
        print(final_response)
        print("-" * 50)


        self.conversation_tree.add_child(parent_id, question, final_response)
        return final_response


    def ask_for_class(self, parent_id, cls):
        """
        Ask a question about the class and handle its properties.
        
        :param parent_id: ID of the parent node in the conversation tree.
        :param cls: The class to explore.
        """
        # Ask a question about the class
        question = get_ontology_question(cls.name, QUESTIONS_DICT)
        answer = self.ask_question(parent_id, question)

        if answer == "N/A":
            return None
        
        # HELP ME HERE: If the answer is a single item, semantically match it with an object in the known ontology and ask_question on it

        # HELP ME HERE: If the answer is a list of items, match each question semantically with an object in the known ontology and ask_question for each

        # HELP ME HERE: If the answer is not semantically related to anything in the ontology, do not add this as a node to the tree

        direct_properties = get_class_properties(self.ontology,cls)


        connected_classes = get_property_class_range(direct_properties)

        # Display the results
        print("Direct properties and connected classes:")
        for prop, classes in connected_classes.items():
            print(f"Property: {prop}, Connected Classes: {classes}")

         # Ask for each directly connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                sub_cls = self.ontology[cls_name]
                self.ask_for_class(parent_id + 1, sub_cls)

        # Create a new node for the class
        class_node_id = self.conversation_tree.node_id


    def start(self):
        """
        Start the questioning process from the base class.
        """
        root_question = """Define the name of the architecture based on the given paper.
            Provide only the name.
            Example names for different papers could be 'GoogleLeNet','ResNet50', 'VGG16'. Only provide the name.
            If no names of that types are provided, respond with "ANNConfiguration".
            """
        
        root_answer = self.ask_question(0, root_question)

        direct_properties = get_class_properties(self.ontology,self.base_class)


        connected_classes = get_property_class_range(direct_properties)

        # Display the results
        print("Direct properties and connected classes:")
        for prop, classes in connected_classes.items():
            print(f"Property: {prop}, Connected Classes: {classes}")

         # Ask for each directly connected class
        for prop, classes in connected_classes.items():
            for cls_name in classes:
                cls = self.ontology[cls_name]
                self.ask_for_class(1, cls)
    

def get_property_class_range(properties):
    connected_classes = {}
    
    for prop in properties:
        connected_classes[prop.name] = [cls.name for cls in prop.range]
    return connected_classes




class ConversationTree:
    """
    A class for managing and navigating a hierarchical conversation tree.
    """

    def __init__(self):
        """
        Initialize the conversation tree with a root node.
        """
        self.tree = {"id": 0, "question": None, "answer": None, "children": []}
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}  # Map of node_id to tree nodes

    def add_child(self, parent_id, question, answer=None):
        """
        Add a child node to a specific parent node in the tree.
        
        :param parent_id: ID of the parent node to attach the child to.
        :param question: Question for the child node.
        :param answer: Optional answer for the child node.
        """
        self.node_id += 1
        child_node = {"id": self.node_id, "question": question, "answer": answer, "children": []}
        self.nodes[parent_id]["children"].append(child_node)
        self.nodes[self.node_id] = child_node

    def to_serializable(self, obj):
        from requests.models import Response

        """
        Convert non-serializable objects to a JSON-compatible structure.
        """
        if isinstance(obj, dict):
            return {key: self.to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self.to_serializable(vars(obj))
        elif isinstance(obj, Response):  # Handle the `Response` object
            return {"response_text": str(obj)}
        else:
            return obj

    def save_to_json(self, file_name):
        """
        Save the conversation tree to a JSON file.

        :param file_name: Path to the file where the tree will be saved.
        """
        with open(file_name, "w") as f:
            json.dump(self.to_serializable(self.tree), f, indent=4)


def load_ontology_questions(json_path: str) -> dict:
    """
    Loads ontology questions from a JSON file.
    :param json_path: Path to the JSON file containing ontology questions.
    :type json_path: str
    :return: Dictionary of ontology questions.
    :rtype: dict
    """
    with open(json_path, 'r') as file:
        questions = json.load(file)
    return questions

def get_ontology_question(entity: str, questions: dict) -> str:
    """
    Retrieves a question for a given entity and type (Classes, ObjectProperties, or DataProperties).
    :param entity: The entity name (e.g., 'Dataset', 'batch_size').
    :type entity: str
    :param entity_type: The type of entity (e.g., 'Classes', 'ObjectProperties', 'DataProperties').
    :type entity_type: str
    :param questions: Dictionary of ontology questions.
    :type questions: dict
    :return: The associated question or a default message if not found.
    :rtype: str
    """
    entity_type = "ObjectPrompts"
    return questions.get(entity_type, {}).get(entity, f"No question found for '{entity}' in '{entity_type}'.")

def get_initial_prompt() -> str:
    initial_prompt = """
                    Work out your chain of thought.
                    If you do not know the answer to a question, respond with "N/A." and nothing else.
                    Question:
                    """
    return initial_prompt

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



# Example usage
if __name__ == "__main__":

    pdf_path = "data/papers/AlexNet.pdf"
    # pdf_path = "data/papers/ResNet.pdf"
    documents = load_pdf(pdf_path)
    chunked_docs = chunk_document(documents)
    embed_model = EmbeddingModel(model_name="all-MiniLM-L6-v2").get_model()
    llm_model = LLMModel(model_name="llama3.1:8b").get_llm()
    indexer = DocumentIndexer(embed_model,llm_model,chunked_docs)
    rag_query_engine = indexer.get_rag_query_engine()

    # Load questions from the ontology JSON file
    json_path = "rag/ontology_prompts.json"
    questions_dict = load_ontology_questions(json_path)
    QUESTIONS_DICT = questions_dict

    # Load ontology
    ontology = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

    # Initialize the conversation tree
    tree = ConversationTree()

    # Initialize the OntologyTreeQuestioner
    questioner = OntologyTreeQuestioner(ontology=ontology, conversation_tree=tree, rag_query_engine=rag_query_engine)

    # Start the questioning process
    questioner.start()

    # Save the conversation tree to JSON
    tree.save_to_json("local_dir/trash_conversation_tree.json")

    
    # # List all classes
    # print("Classes in the ontology:")
    # for cls in ontology.classes():
    #     print(f"- {cls.name}")

    # # List all object properties
    # print("\nObject Properties in the ontology:")
    # for prop in ontology.object_properties():
    #     print(f"- {prop.name}")

    # # List all data properties
    # print("\nData Properties in the ontology:")
    # for prop in ontology.data_properties():
    #     print(f"- {prop.name}")
