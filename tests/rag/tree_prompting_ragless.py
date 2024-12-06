import json
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError

from utils.constants import Constants as C
from utils.llm_model import OllamaLLMModel
from utils.conversational_tree import ConversationTree

from utils.parse_annetto_structure import *
from utils.owl import *


LLM_MODEL_NAME = 'llama3.2:1b'


class LLMResponse(BaseModel):
    instance_names: List[str]  # A list of ontology class names expected from LLM
    
class OntologyTreeQuestioner:
    """
    Generates questions based on ontology classes and their properties,
    integrates with a conversation tree, and recursively asks questions for object properties.
    """

    def __init__(self, ontology=None, conversation_tree=None, llm=None, paper_content:str=None):
        self.ontology = ontology
        self.conversation_tree = conversation_tree
        self.llm = llm
        self.paper_content = paper_content
        self.root_class = self.ontology.ANNConfiguration

    def ask_question(self, parent_id, cls, retries=3):
        for attempt in range(retries):
            try:
                # Retrieve ancestor classes
                ancestor_classes = self.get_ancestor_classes(parent_id)
                print("Ancestor classes:", ancestor_classes)

                # Generate the prompt and question
                class_prompt = generate_class_prompt(cls.name, ancestor_classes, self.paper_content)

                # Query the LLM
                response = self.llm.query_ollama(class_prompt,get_ollama_prompt())

                # Validate and process the response
                validated_response = self.validate_response(response)
                return class_prompt, validated_response

            except (ValueError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Return None if LLM fails to respond with valid JSON
        return None, None

    def handle_class(self, cls, parent_id, visited_classes=None):
        if visited_classes is None:
            visited_classes = set()

        if cls in visited_classes:
            return
        if cls is self.ontology.DataCharacterization:
            return

        visited_classes.add(cls)

        # instantiate data property here?

        requires_instance = requires_final_instantiation(cls, self.ontology)

        question = None
        answer = None

        if requires_instance:
            # Ask the question and get the answer
            question, answer = self.ask_question(parent_id=parent_id, cls=cls)

        # Add the node to the conversation tree with question and answer
        new_node_id = self.conversation_tree.add_child(
            parent_id, cls.name, question=question, answer=answer
        )


        for related_cls in get_connected_classes(cls, self.ontology):
            self.handle_class(related_cls, new_node_id, visited_classes)

        for subcls in get_subclasses(cls):
            self.handle_class(subcls, new_node_id, visited_classes)

    def get_ancestor_classes(self, node_id):
        ancestor_classes = []
        current_id = node_id
        while current_id != 0:
            print("hi")
            node = self.conversation_tree.nodes.get(current_id)
            if not node:
                break
            ancestor_classes.append(node['cls_name'])
            current_id = node.get('parent_id')
        return ancestor_classes[::-1]

    def validate_response(self, response):
        response_json = json.loads(response.strip())
        validated_response = LLMResponse.model_validate(response_json)
        return validated_response

    def start(self):
        if self.root_class is None:
            print(f"Class '{self.root_class.name}' does not exist in the ontology.")
            return        

        visited_classes = set()
        self.handle_class(self.root_class, parent_id=0, visited_classes=visited_classes)

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



def get_CoT_prompt():
    return """
You are a highly capable ontology populator.
Your task is to provide detailed, logical reasoning while answering a question.
If a question requires complex thought, break it down step-by-step to reach a clear conclusion. 
If you do not know the answer or cannot confidently determine it, explicitly state that you do not know and avoid providing incorrect or speculative answers. 

When responding, consider the following:
- Analyze the question logically and methodically.
- Avoid skipping steps in reasoning.
- If unsure, acknowledge your uncertainty honestly. The output to JSON should be 

Respond with clarity, providing step-by-step reasoning or acknowledging uncertainty when needed.
"""

def get_JSON_prompt():
    return """
You are tasked with analyzing a response generated by an LLM in response to a question. 
Your role is to extract the necessary parts of the response and return them as a list of relevant ontology class names in JSON format. 

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
Please respond with a JSON object containing a single key, `instance_names`, which is a list of ontology class names relevant to the question.
Example:
{
    "instance_names": ["ResNet", "VGG16"]
}
"""

def get_ollama_prompt() -> str:
    return """
You are provided with the following things:
1 - A question to answer delimited by triple backticks `.
Your task is to perform the following actions: 
1 - Summarize the following question with details from the provided paper.
2 - Break down the question step-by-step to reach a clear conculsion ot the question.
3 - List each relevant ontology entity names the question asks for.
4 - Output a the list of 

If you do not know the answer or cannot confidently determine it, explicitly state 'I don't know.' and avoid providing incorrect or speculative answers.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in summary>
Output JSON: <json with summary and num_names>

```{query}```
"""

def get_class_context(class_obj):
    class_name = class_obj.name
    ancestor_classes = class_obj.get_ancestors()
    return class_name, ancestor_classes

def generate_class_prompt(class_name, ancestor_classes, paper_content):
    prompt = f"""
    As an expert in neural networks architectures, create a question that

    """
        
        
        
        
    """
    You are a machine that generates questions. Based on the following information:

    - **Class in Ontology**: '{class_name}'
    - **Instantiated Ancestor Classes**: '{', '.join(ancestor_classes) if ancestor_classes else 'None'}'

    Below is a research paper that provides detailed information about the class and its context:
    \"\"\"
    {paper_content}
    \"\"\"

    Please perform the following tasks:
    1. Generate a question that will be used in an llm prompt to instantiate a name for the class.
    2. In the question, relate the class to its ancestor classes where applicable.
    """

    generated_prompt = OllamaLLMModel(model_name=LLM_MODEL_NAME).query_ollama('',prompt)

    print(generated_prompt)


    return generated_prompt

"""
main function
"""


def main():
    from utils.pdf_loader import load_pdf

    file_path = "data/hand_processed/AlexNet.pdf"  # Replace with your actual file path
    doc_str = load_pdf(file_path, as_str=True)
    
    llm = OllamaLLMModel(model_name=LLM_MODEL_NAME)
    
    onto = load_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}")

    tree = ConversationTree()
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        llm=llm,
        paper_content=doc_str
    )
    questioner.start()
    tree.save_to_json("output/conversation_tree.json")

if __name__ == "__main__":
    main()