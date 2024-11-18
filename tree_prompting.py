import json
from owlready2 import *
from typing import List, Dict

from utils.constants import Constants as C

from utils.query_llm import LLM
from utils.pdf_loader import load_pdf
from utils.llm_model import LLMModel
from utils.embedding_model import EmbeddingModel
from utils.document_splitter import DocumentSplitter
from utils.document_indexer import DocumentIndexer

from utils.owl import *


'''
Example usage:
from utils.ConversationTree import ConversationTree
'''

def get_prompt_data_properties(onto_class: ThingClass, property: Property) -> str:
    prompt = f'Does {onto_class} have the data property {property.name}, if so what is its value'
    return prompt

def get_prompt_class(onto_class: ThingClass) -> str:
    prompt = f'Does this have an {onto_class}'
    return prompt

def get_prompt_object_property(onto_class: ThingClass, property: Property) -> str:
    prompt = f'Does {onto_class} have {property}'
    return prompt
    
class OntologyTreeQuestioner:
    """
    A class to generate questions based on ontology classes and their properties,
    integrate with a conversation tree, and recursively ask questions for object properties.
    """

    def __init__(self, ontology, base_class, conversation_tree, llm=None):
        """
        Initialize with ontology, base class, and conversation tree.
        
        :param ontology: Loaded ontology object.
        :param base_class: Base class to start questioning.
        :param conversation_tree: Instance of the ConversationTree class.
        """
        self.ontology = ontology
        self.base_class = base_class
        self.conversation_tree = conversation_tree
        self.llm=llm

    def ask_question(self, parent_id, question):
        """
        Asks question to LLM
        
        :param parent_id: ID of the parent node in the conversation tree.
        :param question: The question to ask.
        :return: Mock answer as a string.
        """

        prompt_template = """prompt_template =
        Please answer the following question concisely and in list format when applicable. Do not provide any additional context or verbose explanations. If the answer involves steps, points, or attributes, present them as a structured list. Otherwise, provide an atomic and precise response.
        Question: {question}
        Answer:
        """
        prompt = prompt_template.format(question=question)
        print(f"Question: {question}")
        answer = llm.query(question)

        self.conversation_tree.add_child(parent_id, question, answer)
        return answer


    def handle_property(self, parent_id, prop):
        """
        Handle a property by determining its type and recursively generating questions.
        
        :param parent_id: ID of the parent node in the conversation tree.
        :param prop: The property to handle.
        """
        print(f"\nHandling property: {prop.name}")

        range_type = get_property_range_type(prop)

        if range_type == ' class':
            # Ask a question about the property and explore its range
            question = f"What is the relationship of '{prop.name}'?"
            answer = self.ask_question(parent_id, question)

            # Recursively handle classes in the range
            for range_cls in prop.range:
                self.ask_for_class(parent_id, range_cls)
        elif range_type == 'atomic':
            # Ask a question about the atomic property value
            question = f"What is the value of '{prop.name}'?"
            answer = self.ask_question(parent_id, question)
        else:
            print(f"Property '{prop.name}' is neither an ObjectProperty nor a DataProperty.")

    def ask_for_class(self, parent_id, cls):
        """
        Ask a question about the class and handle its properties.
        
        :param parent_id: ID of the parent node in the conversation tree.
        :param cls: The class to explore.
        """
        # Ask a question about the class
        question = f"What can you tell me about the class '{cls.name}'?"
        answer = self.ask_question(parent_id, question)

        # Create a new node for the class
        class_node_id = self.conversation_tree.node_id

        # Iterate over properties of the class
        properties = get_class_properties(self.ontology,cls)
        for prop in properties:
            self.handle_property(class_node_id, prop)

    def start(self):
        """
        Start the questioning process from the base class.
        """
        print(f"Starting from base class: {self.base_class.name}")
        root_question = (
            """Define the name of the name of the architecture for the class 'ANNConfiguration' based on the given paper.
             Provide only the name of a specific architecture for the class
            'ANNConfiguration'. Examples names for different papers could be 'GoogleLeNet','ResNet50', or 'VGG16'. Only provide the name.
            """
        )
        root_answer = self.ask_question(0, root_question)
        self.ask_for_class(0, self.base_class)


class ConversationTree:
    """
    A class for managing and navigating a hierarchical conversation tree.
    This tree structure can store questions, answers, and their relationships, 
    enabling conversational context tracking.
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

    def save_to_json(self, file_name):
        """
        Save the conversation tree to a JSON file.

        :param file_name: Path to the file where the tree will be saved.
        """
        with open(file_name, "w") as f:
            json.dump(self.tree, f, indent=4)

#Look into this

def parse_to_list(answer):
    """
    Converts a verbose answer to a list format.

    :param answer: The raw answer string from the LLM.
    :return: A list of answers or a single atomic response.
    """
    # Split by newline or bullets to extract list items
    return [line.strip() for line in answer.split('\n') if line.strip()]


# Example usage
if __name__ == "__main__":

    #Loads PDF to 
    documents = load_pdf("data/papers/AlexNet.pdf")

    splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split(documents)

    embed_model = EmbeddingModel().get_model()
    llm_predictor = LLMModel().get_llm()

    indexer = DocumentIndexer(embed_model, llm_predictor)
    vector_index = indexer.create_index(split_docs)

    llm = LLM(vector_index,llm_predictor)

    # Load ontology
    ontology = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

    # Get base class
    base_class = ontology.ANNConfiguration
    print(f"Base class: {base_class.name}")

    # List all classes
    print("Classes in the ontology:")
    for cls in ontology.classes():
        print(f"- {cls.name}")

    # List all object properties
    print("\nObject Properties in the ontology:")
    for prop in ontology.object_properties():
        print(f"- {prop.name}")

    # List all data properties
    print("\nData Properties in the ontology:")
    for prop in ontology.data_properties():
        print(f"- {prop.name}")

    # Initialize the conversation tree
    tree = ConversationTree()

    # Initialize the OntologyTreeQuestioner
    questioner = OntologyTreeQuestioner(ontology=ontology, base_class=base_class, conversation_tree=tree, llm=llm)

    # Start the questioning process
    questioner.start()

    # Save the conversation tree to JSON
    tree.save_to_json("local_dir/trash_conversation_tree.json")


# class ConversationTree:
#     """
#     A class for managing and navigating a hierarchical conversation tree.
#     This tree structure can store questions, answers, and their relationships, 
#     enabling conversational context tracking.

#     Attributes:
#         tree (dict): The root node of the tree.
#         node_id (int): A counter for generating unique IDs for nodes.
#         current_node (dict): The current node in the tree.
#         nodes (dict): A mapping of node IDs to nodes.
#     """

#     def __init__(self):
#         """
#         Initialize the conversation tree with a root node.
#         """
#         self.tree = {"id": 0, "question": None, "answer": None, "children": []}
#         self.node_id = 0
#         self.current_node = self.tree
#         self.nodes = {0: self.tree}  # Map of node_id to tree nodes

#     def add_child(self, parent_id, question, answer=None):
#         """
#         Add a child node to a specific parent node in the tree.
        
#         :param parent_id: ID of the parent node to attach the child to.
#         :type parent_id: int
#         :param question: Question for the child node.
#         :type question: str
#         :param answer: Optional answer for the child node.
#         :type answer: str
#         """
#         self.node_id += 1
#         child_node = {"id": self.node_id, "question": question, "answer": answer, "children": []}
#         self.nodes[parent_id]["children"].append(child_node)
#         self.nodes[self.node_id] = child_node

#     def update_node_answer(self, node_id, answer):
#         """
#         Update the answer for a specific node in the tree.

#         :param node_id: ID of the node to update.
#         :type node_id: int
#         :param answer: New answer for the node.
#         :type answer: str
#         """
#         if node_id in self.nodes:
#             self.nodes[node_id]["answer"] = answer

#     def get_ancestor_context(self, node_id):
#         """
#         Retrieve the context of ancestor questions and answers for a specific node.
        
#         :param node_id: ID of the node for which to retrieve the context.
#         :type node_id: int
#         :return: A list of dictionaries containing ancestor questions and answers.
#         :rtype: list
#         """
#         context = []
#         current_node = self.nodes[node_id]
#         while current_node.get("id") != 0:  # Stop at the root node
#             context.insert(0, {
#                 "question": current_node["question"],
#                 "answer": current_node["answer"]
#             })
#             parent_node = next((node for node in self.nodes.values()
#                                 if current_node in node["children"]), None)
#             if parent_node:
#                 current_node = parent_node
#             else:
#                 break
#         return context

#     def save_to_json(self, file_name):
#         """
#         Save the conversation tree to a JSON file.

#         :param file_name: Path to the file where the tree will be saved.
#         :type file_name: str
#         """
#         with open(file_name, "w") as f:
#             json.dump(self.tree, f, indent=4)

#     def load_from_json(self, file_name):
#         """
#         Load the conversation tree from a JSON file.

#         :param file_name: Path to the JSON file to load.
#         :type file_name: str
#         """
#         with open(file_name, "r") as f:
#             self.tree = json.load(f)
#             self._rebuild_node_map(self.tree)

#     def _rebuild_node_map(self, node):
#         """
#         Rebuild the internal node map from a loaded tree structure.

#         :param node: The current node in the tree structure.
#         :type node: dict
#         """
#         self.nodes = {}
#         self.node_id = 0
#         self._rebuild_helper(node)

#     def _rebuild_helper(self, node):
#         """
#         Helper function to recursively rebuild the node map.

#         :param node: The current node to process.
#         :type node: dict
#         """
#         self.nodes[node["id"]] = node
#         self.node_id = max(self.node_id, node["id"])
#         for child in node["children"]:
#             self._rebuild_helper(child)

# class OntologyTreeQuestioner:
#     """
#     A class to generate questions based on ontology classes and their properties.
#     Recursively asks questions for object properties or terminates with atomic property values.
#     """

#     def __init__(self, ontology, base_class):
#         """
#         Initialize with ontology and base class.
        
#         :param ontology: Loaded ontology object.
#         :param base_class: Base class to start questioning.
#         """
#         self.ontology = ontology
#         self.base_class = base_class
#         self.questions = []

#     def ask_question(self, question: str):
#         """
#         Simulate asking a question and returning an answer (for demonstration).
#         Replace this with an LLM or user interaction as needed.
        
#         :param question: The question to ask.
#         :return: Mock answer as a string.
#         """
#         print(f"Question: {question}")
#         return "Mock answer for demonstration"

#     def get_class_properties(self, cls):
#         """
#         Get all properties associated with a class.
        
#         :param cls: The class to inspect.
#         :return: List of properties.
#         """
#         return list(cls.get_properties())

#     def handle_property(self, prop):
#         """
#         Handle a property by determining its type and recursively generating questions.
        
#         :param prop: The property to handle.
#         """
#         print(f"\nHandling property: {prop.name}")

#         range_type = get_property_range_type(prop)

#         if range_type == 'class':
#             # Ask a question about the property and explore its range
#             question = f"What is the relationship of '{prop.name}'?"
#             answer = self.ask_question(question)

#             # Recursively handle classes in the range
#             for range_cls in prop.range:
#                 self.ask_for_class(range_cls)
        
#         elif range_type == 'atomic':
#             # Ask a question about the atomic property value
#             question = f"What is the value of '{prop.name}'?"
#             answer = self.ask_question(question)
#         else:
#             print(f"Property '{prop.name}' is neither an ObjectProperty nor a DataProperty.")

#     def ask_for_class(self, cls):
#         """
#         Ask a question about the class and handle its properties.
        
#         :param cls: The class to explore.
#         """
#         # Ask a question about the class
#         question = f"What can you tell me about the class '{cls.name}'?"
#         answer = self.ask_question(question)

#         # Iterate over properties of the class
#         properties = self.get_class_properties(cls)
#         for prop in properties:
#             self.handle_property(prop)

#     def start(self):
#         """
#         Start the questioning process from the base class.
#         """
#         print(f"Starting from base class: {self.base_class.name}")
#         self.ask_for_class(self.base_class)


# # Example usage
# if __name__ == "__main__":
#     # Load ontology
#     onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

#     # Get base class
#     base_class = onto.ANNConfiguration
#     print(f"Base class: {base_class.name}")

#     # Initialize the OntologyTreeQuestioner
#     questioner = OntologyTreeQuestioner(ontology=onto, base_class=base_class)

#     # Start the questioning process
#     questioner.start()


# # Example usage
# if __name__ == "__main__":
#     """
#     Demonstrates how to use the ConversationTree class for conversational context tracking.

#     - Initialize a conversation tree.
#     - Interact with an LLM to generate answers for questions.
#     - Save and load the conversation tree as JSON.
#     """

#     # Initialize conversation tree
#     tree = ConversationTree()


    

#     # Load ontology
#     onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()

#     #Get base class
#     base_class = onto.ANNConfiguration
#     print(f"Base class: {base_class.name}")

#     # List all classes in the ontology
#     for cls in onto.classes():
#         break
#         print(f"Class: {cls.name}")

#     #Get all properties to a class
#     class_properties = None
#     class_properties = get_class_properties(onto,base_class)
#     print(class_properties)

#     possible_classes = None

#     # for property in class_properties:
#     #     possible_classes 

#     # Define a function to get all properties of a class

# # Get all properties of the base class
# class_properties = get_class_properties(onto,base_class)

# # Iterate over each property and print details
# for prop in class_properties:
#     print(f"\nProperty: {prop.name}")

#     range_type = get_property_range_type(prop)
    
#     # Check if it is an ObjectProperty or DataProperty
#     if range_type == 'class':
#         print("  Type: ObjectProperty")
        
#         # Print possible classes (ranges) associated with the property
#         ranges = prop.range
#         if ranges:
#             print("  Possible Classes (Range):")
#             for r in ranges:
#                 print(f"    - {r.name}")
#         else:
#             print("  Possible Classes (Range): None")

#     elif range_type == 'atomic':
#         print("  Type: DataProperty")
        
#         # Print the data range if specified
#         ranges = prop.range
#         if ranges:
#             print("  Data Range:")
#             for r in ranges:
#                 print(f"    - {r}")
#         else:
#             print("  Data Range: None")
#     else:
#         print("  Type: Unknown property type")




#     # # Root question
#     # root_question = "What networks are in a GAN?"
#     # root_answer = query_llm(root_question)
#     # tree.add_child(0, root_question, root_answer)

#     # # Add child questions
#     # parent_id = 1  # ID of the root node
#     # child_questions = [
#     #     "How many layers does the generator network have?",
#     #     "How many layers does the adversarial network have?"
#     # ]
#     # for question in child_questions:
#     #     context = tree.get_ancestor_context(parent_id)
#     #     answer = query_llm(question, context)
#     #     tree.add_child(parent_id, question, answer)

#     # # Save conversation to JSON
#     # tree.save_to_json("local_dir/test_conversation_tree.json")

