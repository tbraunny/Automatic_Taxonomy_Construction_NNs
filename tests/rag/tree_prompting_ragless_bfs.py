import json
import re
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError

from utils.constants import Constants as C
from utils.llm_model import OllamaLLMModel
from utils.conversational_tree import ConversationTree
from utils.parse_annetto_structure import *
from utils.owl import *


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
            parent_id (int): ID of the parent node in the conversation tree.
            cls: The ontology class to generate the question for.
            retries (int): Number of retry attempts in case of failures.

        Returns:
            tuple: A tuple containing the generated question (str) and class names (list).
        """
        for attempt in range(retries):
            try:
                # Prepare the class question
                class_question = f"What are the {cls.name}(s) in the given context?"

                

                # return class_question, [f"{cls.name}_instance_1",f"{cls.name}_instance_2"]

                class_question_ctx = class_question + f"Paper context: '''{self.paper_content}'''"

                instructions = get_ollama_instructions()

                # Query the LLM
                print(f"Querying on class: {cls.name}...\n")
                response = self.llm.query_ollama(class_question_ctx, instructions)

                # Extract and validate the response JSON
                JSON_response = extract_JSON(response)
                validated_response = self.validate_response(JSON_response)

                # Move on if there are no instances of cls
                if validated_response.instance_names == []:
                    return
                
                print(f"Question: {class_question}\n Instances: {validated_response.instance_names}\n")

                return class_question, validated_response.instance_names

            except (ValueError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        return None

    def handle_class(self, root_cls, parent_id):
        """
        Recursively handles an ontology class and its related classes/subclasses.

        Args:
            root_cls: The root class to start processing.
            parent_id (int): ID of the parent node in the conversation tree.
        """
        visited_classes = set()
        queue = [(root_cls, parent_id)]

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
            queue.extend([(related_cls, new_node_id) for related_cls in get_connected_classes(cls, self.ontology) if related_cls not in visited_classes])
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
        root_class = self.ontology.ANNConfiguration
        if root_class is None:
            print("Class 'Network' does not exist in the ontology.")
            return

        self.handle_class(root_class, parent_id=0)


def main():
    """
    Main entry point for initializing and running the ontology questioner.
    """
    llm, paper_content = use_ollama_model()
    onto = load_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}")
    tree = ConversationTree()

    questioner = OntologyTreeQuestioner(ontology=onto, conversation_tree=tree, llm=llm, paper_content=paper_content)
    questioner.start()
    tree.save_to_json("output/conversation_tree.json")


def extract_JSON(response: str) -> dict:
    """
    Extracts JSON data from a response string.

    Args:
        response (str): The LLM's response containing JSON data.

    Returns:
        dict: Extracted JSON object.

    Raises:
        ValueError: If no valid JSON block is found in the response.
    """
    try:
        json_match = re.search(r'```json\n({.*?})\n```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError("No valid JSON block found in the response.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}\nResponse: {response}")


def use_ollama_model():
    """
    Loads the LLM model and paper content for ontology processing.

    Returns:
        tuple: A tuple containing the LLM model and the paper content.
    """
    from utils.pdf_loader import load_pdf
    file_path = "data/hand_processed/AlexNet.pdf"
    documents = load_pdf(file_path)
    paper_content = "\n".join([doc.page_content for doc in documents])
    llm = OllamaLLMModel(temperature=0.5, top_k=7)
    return llm, paper_content


def get_ollama_instructions():
    prompt = """
You are provided with the following things:
1 - A question to answer delimited by triple backticks `.
2 - Previously, a paper as context written about a neural network architecture.
Your task is to perform the following actions: 
1 - Summarize the following question with details from the provided paper.
2 - Break down the question step-by-step to reach a clear conclusion to the question.
3 - List each relevant ontology entity names the question asks for.
4 - Output a JSON object of ontology entity names that contains the following key: instance_names.

If you do not know the answer or cannot confidently determine it, explicitly state 'I don't know.' and avoid providing incorrect or speculative answers.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Names: <list of names in summary>
Output JSON: <json with ontology entity names>

Example JSON Format:
```json
{{
    "instance_names": ["ResNet", "VGG16"]
}}
{{
    "instance_names": ["Convolutional Layer 1", "Convolutional Layer 2"]
}}
"""
    prompt += """```{query}```"""
    return prompt


def generate_class_question(class_name: str, ancestor_class_names: list,llm) -> str:
    """
    Generates a detailed prompt for ontology population using a given class name
    and its ancestor classes. The prompt is designed to guide an LLM in naming
    the class and relating it to its ancestors, within the context of neural
    network architectures.

    Args:
        class_name (str): The name of the ontology class to generate the question for.
        ancestor_class_names (list): A list of ancestor class names for the given class.

    Returns:
        str: A formatted string containing the generated prompt.
    """
    # Create a string representation of ancestor classes:
    # If ancestor_class_names is not empty, join the names with a comma and space.
    # Otherwise, use 'None' to indicate no ancestors are provided.
    ancestors = ', '.join(ancestor_class_names) if ancestor_class_names else 'None'
    
    prompt = f"""
You are an expert in ontology population, specializing in neural networks.
The ontology is being built based on a variety of research papers that describe, analyze,
and propose different aspects of neural networks. These include architectures (e.g., CNNs, RNNs, Transformers, GANs), 
components (e.g., layers, activation functions, optimizers), methods, and other related elements.

- **Class**: {class_name}
- **Ancestor Classes**: {ancestors}

Your task is to perform the following:
1. Generate a concise question that will guide an LLM in identifying the most appropriate name or list of names for the given class.
2. Do not include any additional context, definitions, or characteristics about the class in the question.
3. Ensure the question is focused solely on asking for a name or list of names for the class.

Output the question as a JSON object using the key 'question'.

Output JSON: <json with question>

Example JSON Format:
{{
    "question": "What is the name of this *insert class name*?"
}}
Example JSON Format:
{{
    "question": "What are the names associated with this *insert class name* in the *insert ancestor network name*?"
}}

The generated question should be simple, clear, and relevant to the naming of the class.
"""

    print('Generating question...')
    response = llm.query_ollama(prompt, '  {query}  ')
    json_data = extract_JSON(response)
    key = 'question'
    if key in json_data:
        return json_data[key]
    else:
        raise ValueError("The key 'question' was not found in the JSON object.")
    

if __name__ == "__main__":
    main()
