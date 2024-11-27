import json
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError
from utils.constants import Constants as C
from utils.query_rag import RemoteDocumentIndexer
from utils.conversational_tree import ConversationTree
from utils.parse_annetto_structure import *
from utils.owl import *




class LLMResponse(BaseModel):
    instance_names: List[str]  # A list of ontology class names expected from LLM



class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology, conversation_tree, query_engine, onto_prompts):
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
        self.onto_prompts=onto_prompts # Unused and need to be rewritten

    def ask_question(self, parent_id, question, cls, retries=3):
        for attempt in range(retries):
            try:
                # Retrieve ancestor context
                ancestor_context = self.get_ancestor_context(parent_id)

                if ancestor_context:
                    print(f"Ancestor context:\n{ancestor_context}\n") # For debugging ***************

                # Get the chain-of-thought prompt
                chain_of_thought_prompt = get_chain_of_thought_prompt()

                # Get few-shot examples if available
                few_shot_examples = get_few_shot_examples(cls, self.onto_prompts) # Needs to be reimplemented along with new prompts ***********

                # Build the CoT prompt
                prompt = build_prompt(
                    instructions=chain_of_thought_prompt,
                    context=ancestor_context,
                    question=question,
                    few_shot_examples=few_shot_examples
                )

                # Query the LLM to get a more thouhgt out and correct answer
                cot_response = self.llm.query(prompt)

                # Build the JSON extraction prompt
                json_prompt = f"{get_json_prompt()}\n\nQuestion:\n{question}\n\nResponse:\n{cot_response}\n\nYour JSON response:"

                # Query the LLM for the final JSON response
                response = self.llm.query(json_prompt)
                
                if response.strip() == 'N/A': # Need better logic for filtering 'i don't know' answers ***************
                    return None

                # Pydantic library validates JSON format
                validated_response = LLMResponse.parse_raw(response.strip()) # parse_raw() is depricated and needs to be updated *********

                return validated_response

            except (ValueError, ValidationError) as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Return None if Llama fails to response in valididated JSON
        return None
    
    

    def handle_class(self, cls, parent_id, visited_classes=None):
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

        # Create a node for the current class
        new_node_id = self.conversation_tree.add_child(parent_id, cls.name, answer=None)

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

    def get_ancestor_context(self, node_id):
        """
        Collects ancestor answers up to the given node and formats them as a string to provide context.

        :param node_id: The ID of the current node.
        :return: A formatted string of questions and their corresponding answers, or an empty string if no context exists.
        """
        answers = []
        current_id = node_id
        while current_id is not None:
            node = self.conversation_tree.nodes.get(current_id)
            if not node:
                break

            if node['answer']:
                # node['answer'] is a dict with questions as keys and instance names as values
                for question_text, instance_names in node['answer'].items():
                    if instance_names:
                        # Ensure instance_names is a list
                        if not isinstance(instance_names, list):
                            instance_names = [instance_names]
                        answers.append((question_text, instance_names))

            current_id = node.get('parent_id')  # Move to parent node

        if not answers:
            # If no context, return empty string
            return ""

        # Format the context as a string
        context_str = "\n".join(
            [f"Step {i + 1}: Question: {q}\nAnswer: {', '.join(a)}"
            for i, (q, a) in enumerate(reversed(answers))]
        )
        return context_str



    def start(self):
        """
        Starts the questioning process from the base class.
        """
        root_class = self.ontology.Network 

        if root_class is None:
            print(f"Class '{root_class.name}' does not exist in the ontology.")
            return

        visited_classes = set()

        # Start processing from the root class
        self.handle_class(root_class, parent_id=0, visited_classes=visited_classes)



def main():
    query_engine = RemoteDocumentIndexer('100.105.5.55',5000).get_rag_query_engine()
    # response = query_engine.query("hello llama")
    # print(response)
    # return
    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
    onto_prompts_json = load_ontology_questions("rag/tree_prompting/ontology_prompts.json")   


    tree = ConversationTree()
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        query_engine=query_engine,
        onto_prompts=onto_prompts_json
    )
    questioner.start()
    tree.save_to_json("./rag/tree_prompting/output/conversation_tree.json")



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

def build_prompt(instructions, question, context=None, few_shot_examples=None):
    """
    Builds a structured prompt for the LLM.

    :param chain_of_thought_prompt: Instructions for the LLM to follow.
    :param context: Ancestor context to provide background information.
    :param question: The question to be answered.
    :param few_shot_examples: Optional few-shot examples for guidance.
    :return: A well-formatted prompt string.
    """

    prompt_parts = [instructions.strip()]

    if context:
        prompt_parts.append(f"Context:\n{context.strip()}")

    if few_shot_examples:
        prompt_parts.append(f"Few-Shot Examples:\n{few_shot_examples.strip()}")

    prompt_parts.append(f"Question:\n{question.strip()}")

    prompt = "\n\n".join(prompt_parts)
    return prompt



def get_chain_of_thought_prompt():
    return '''
You are a highly capable and thoughtful language model. Your task is to provide detailed, logical reasoning while answering a question. If a question requires complex thought, break it down step-by-step to reach a clear conclusion. 
If you do not know the answer or cannot confidently determine it, explicitly state that you do not know and avoid providing incorrect or speculative answers. 
When responding, consider the following:
- Analyze the question logically and methodically.
- Avoid skipping steps in reasoning.
- If unsure, acknowledge your uncertainty honestly and suggest next steps for finding the answer.
Respond with clarity, providing step-by-step reasoning or acknowledging uncertainty when needed.
'''

"""
Work out your chain of thought.
If you do not know the answer to a question, respond with "N/A" and nothing else.
"""

def get_json_prompt():
    return """
You are tasked with analyzing a response generated by an LLM in response to a question. Your role is to extract the necessary parts of the response and return them as a list of relevant ontology class names in JSON format. 

Specifically:
- Review the original question and the provided response.
- Identify terms or concepts in the response that correspond to ontology class names relevant to the question.
- Exclude unrelated text or extraneous details.
- Return your output as a JSON object with a single key: `instance_names`, which contains a list of relevant ontology class names.
Expected output format:
{
    "instance_names": ["ResNet", "VGG16"]
}
Ensure the list is accurate and succinct, focusing only on directly relevant ontology class names.
"""

"""
You have been given a question and its correct response. 
Please respond with a JSON object containing a single key, `instance_names`, 
which is a list of ontology class names relevant to the question.
Example:
{
    "instance_names": ["ResNet", "VGG16"]
}"""



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


def get_few_shot_examples(entity: str, ontology_data: List[dict]) -> str:
    """
    Retrieves few-shot examples based on the current question.

    :param question: The current question being asked.
    :return: Formatted few-shot examples or None if not available.
    """
    for entry in ontology_data:
        if entry.get("class") == entity:
            # Get the few-shot examples
            examples = entry.get("few_shot_examples", [])
            
            if not examples:
                return None
            # Format the examples
            formatted_examples = "\n".join([
                f"Example {idx + 1}:\n  Question: {ex['input']}\n  Answer: {ex['output']}"
                for idx, ex in enumerate(examples)
            ])
            return formatted_examples.strip()

    return None


if __name__ == "__main__":
    main()
