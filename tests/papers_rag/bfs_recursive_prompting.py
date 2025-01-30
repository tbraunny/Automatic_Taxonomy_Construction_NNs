import json
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError

from utils.constants import Constants as C
from utils.owl.parse_annetto_structure import *
from utils.owl.owl import *


class LLMResponse(BaseModel):
    """
    Data model for validating LLM responses.
    Attributes:
        instance_names (List[str]): List of ontology class names extracted from the LLM's response.
    """
    instance_names: List[str]


class OntologyTreeQuestioner:
    """
    Handles ontology population by generating and asking questions about ontology classes, 
    and organizing responses into a conversation tree.
    """

    def __init__(self, ontology=None, conversation_tree=None, llm=None, paper_content=None):
        """
        Initialize the OntologyTreeQuestioner with necessary components.

        Args:
            ontology: The ontology object to process.
            conversation_tree: The conversation tree for storing questions and answers.
            llm: The LLM model for generating and answering questions.
            paper_content: Content of the reference paper for context.
        """
        self.ontology = ontology
        self.conversation_tree = conversation_tree
        self.llm = llm
        self.paper_content = paper_content

    def ask_question(self, cls, retries=3):
        """
        Generates and asks a question about a given ontology class using an LLM.

        Args:
            cls: The ontology class to generate the question for.
            retries (int): Number of retry attempts in case of failures.

        Returns:
            list of class names strings
        """
        for attempt in range(retries):
            try:
                # Grab prompt for cls
                cls_prompt = f"What are the names of {cls.name}(s)?"

                # append context

                # append json output instructions 

                # Query the LLM
                print(f"Querying on class: {cls.name}...")
                ## Code for querying llm

                # Extract and validate the response JSON
                # JSON_response = extract_JSON(response)
                # validated_response = self.validate_response(JSON_response)

                # Move on if there are no instances of cls
                # if validated_response.instance_names == []:
                #     return
                
                return [f"instance1_{cls.name}",f"instance2_{cls.name}",f"instance3_{cls.name}"]
                # return validated_response.instance_names

            except (ValueError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

    def handle_class(self, cls: ThingClass, visited_classes: set):
        """
        Recursively handles an ontology class and its related classes/subclasses.

        Args:
            root_cls: The root class to start processing.
        """
        queue = [(cls, parent_id)]

        while queue:
            cls, parent_id = queue.pop(0)

            if cls in visited_classes or cls is self.ontology.DataCharacterization:
                continue

            visited_classes.add(cls)
            new_node_id = self.conversation_tree.add_child(parent_id, cls.name, answer=None)

            if requires_final_instantiation(cls, self.ontology):
                result = self.ask_question(cls=cls)
                if result:
                    class_question, class_names = result
                    self.conversation_tree.nodes[new_node_id]['question'] = class_question
                    self.conversation_tree.nodes[new_node_id]['answer'] = class_names

            # Enqueue connected classes and subclasses
            if any(get_connected_classes(cls, self.ontology)):
                queue.extend([(related_cls, new_node_id) for related_cls in get_connected_classes(cls, self.ontology) if related_cls not in visited_classes])
            if any(get_subclasses(cls)):
                queue.extend([(subcls, new_node_id) for subcls in get_subclasses(cls) if subcls not in visited_classes])

  
    def validate_response(self, response:str):
        """
        Validates and parses the JSON response.

        Args:
            response (str): JSON response string.

        Returns:
            LLMResponse: Validated response object.
        """
        response_json = response if isinstance(response, dict) else json.loads(response.strip())
        return LLMResponse.model_validate(response_json)

    def start(self):
        """
        Starts processing the ontology by handling the root class.
        """
        # Ensure ANNConfiguration exists in the ontology
        if not hasattr(self.ontology, 'ANNConfiguration'):
            print("Error: Class 'ANNConfiguration' not found in ontology.")
            return

        start_class = self.ontology.ANNConfiguration
        visited_classes = set()
        self.handle_class(start_class, visited_classes)
        print("Ontology structure written successfully.")

if __name__ == "__main__":
    """
    Main entry point for initializing and running the ontology questioner.
    """
    ontology = load_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}")

    OntologyTreeQuestioner(ontology=ontology).start()