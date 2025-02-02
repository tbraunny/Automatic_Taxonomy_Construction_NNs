from utils.owl_utils import *
from utils.annetto_utils import requires_final_instantiation

from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError

from utils.constants import Constants as C


class LLMResponse(BaseModel):
    instance_names: List[str]  # A list of ontology class names expected from LLM



class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology):
        """
        Initializes the OntologyTreeQuestioner.

        :param ontology: Loaded ontology object.
        :param conversation_tree: Instance of the ConversationTree class.
        :param rag_query_engine: The LLM query engine for querying the language model.
        """
        self.ontology = ontology
        self.base_class = get_base_class(ontology)

    def ask_question(self, parent_id, question, cls, retries=3):
        for attempt in range(retries):
            try:
                # Retrieve ancestor context
                ancestor_context = self.get_ancestor_context(parent_id)

                if ancestor_context:
                    print(f"Ancestor context:\n{ancestor_context}\n") # For debugging ***************

                # Get the chain-of-thought prompt

                # Get few-shot examples if available

                # Build the CoT prompt
                

                # Query the LLM to get a more thouhgt out and correct answer

                # Build the JSON extraction prompt

                # Query the LLM for the final JSON response
                
                # Pydantic library validates JSON format
                validated_response = LLMResponse.parse_raw(response.strip()) # parse_raw() is depricated and needs to be updated *********

                return validated_response

            except (ValueError, ValidationError) as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Return None if Llama fails to response in valididated JSON
        return None
    
    

    def handle_class(self, cls, level, visited_classes=None):
        """
        Handles a class by possibly instantiating it and processing its connected classes.

        :param cls: The class to handle.
        :param parent_id: ID of the parent node in the conversation tree.
        :param visited_classes: Set of classes that have already been visited.
        """

        # Initialize visited_classes if it's None
        if visited_classes is None:
            visited_classes = set()

        # Check if the class has already been visited
        if cls in visited_classes:
            return
        if cls is self.ontology.DataCharacterization: # Junk class for the time being *********
            return
        # Add the current class to the visited set
        visited_classes.add(cls)

        # Check if cls requires final instantiation (Has no object or data properies)
        requires_instance = requires_final_instantiation(cls,self.ontology)


        # Instantiate cls if it has or subclasses or if it requires_instance(no data or object properties)
        if requires_instance or not get_subclasses(cls): # Need logic for instantiating data properties ****************
            question = f'What {cls.name} does this architecture have defined?'

            # Ask the question
            response = self.ask_question(parent_id=new_node_id, question=question, cls=cls)

            if response: # Debug print
                print(f"Question:\n{question}\nResponse:\n{response.instance_names}")

            if response and response.instance_names:
                # Transform instance_names into a dictionary with the class name as the key
                response_dict = {question: response.instance_names}

                # Update the node's answer
                self.conversation_tree.nodes[new_node_id]['answer'] = response_dict

        # Process related classes by object properties
        for related_cls in get_connected_classes(cls, self.ontology):
            self.handle_class(related_cls,new_node_id,visited_classes)

        # Process subclasses of cls
        for subcls in get_subclasses(cls):
            self.handle_class(subcls,new_node_id,visited_classes)



    def start(self):
        """
        Starts the questioning process from the base class.
        """
        root_class = self.ontology.ANNConfiguration # Starting at Network for simplicity and testing

        if root_class is None:
            print(f"Class '{root_class.name}' does not exist in the ontology.")
            return

        visited_classes = set()

        # Start processing from the root class
        self.handle_class(root_class, parent_id=0, visited_classes=visited_classes)



def main():

    # query_engine = LocalRagEngine().get_rag_query_engine()
    # response = query_engine.query("hello llama")
    # print(response)
    # return

    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
    onto_prompts_json = load_ontology_questions("rag/tree_prompting/test_ontology_prompts.json")   


    tree = ConversationTree()
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        query_engine=None,
        onto_prompts=onto_prompts_json
    )
    questioner.start()
    tree.save_to_json("output/conversation_tree.json")

if __name__ == "__main__":
    main()
