import json
from owlready2 import *
from typing import List
from pydantic import BaseModel, ValidationError

from utils.constants import Constants as C
from utils.llm_model import OllamaLLMModel
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

    def __init__(self, ontology=None, conversation_tree=None, llm=None, paper_content=None):
        self.ontology = ontology
        self.conversation_tree = conversation_tree
        self.llm = llm
        self.paper_content=paper_content

    def ask_question(self, parent_id, cls, retries=3):
        for attempt in range(retries):
            try:
                # Retrieve ancestor classes
                # ancestor_classes = self.get_ancestor_classes(parent_id)

                # Generate the prompt
                # prompt = generate_prompt(cls.name, ancestor_classes,self.paper_content)
                prompt = f'{cls.name}'

                place_holder_instructions =f"""Give instance names to the ontology class delimited by triple backticks given the paper \"\"\"{self.paper_content}\"\"\" """
                place_holder_instructions += '```{query}```'
                # Query the LLM
                print(f"Querying on class {cls.name}...")
                response = self.llm.query_ollama(prompt,place_holder_instructions)
                # Validate and process the response
                validated_response = response
                print(f"Response:\n{validated_response}\n")
                # validated_response = self.validate_response(response)
                return validated_response

            except (ValueError, ValidationError) as e:
                print(f"Validation error on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")

        # Return None if LLM fails to respond with valid JSON
        return None

    def handle_class(self, root_cls, parent_id):
        visited_classes = set()
        queue = [(root_cls, parent_id)]
        while queue:
            cls, parent_id = queue.pop(0)  # Pop from front of the queue for BFS

            if cls in visited_classes:
                continue
            if cls is self.ontology.DataCharacterization:
                continue

            visited_classes.add(cls)

            requires_instance = requires_final_instantiation(cls, self.ontology)
            new_node_id = self.conversation_tree.add_child(parent_id, cls.name, answer=None)

            if requires_instance:
                # Ask the question without hardcoding
                response = self.ask_question(parent_id=new_node_id, cls=cls)
                if response and response.instance_names:
                    response_dict = {'instance_names': response.instance_names}
                    self.conversation_tree.nodes[new_node_id]['answer'] = response_dict

            # Enqueue connected classes and subclasses
            for related_cls in get_connected_classes(cls, self.ontology):
                if related_cls not in visited_classes:
                    queue.append((related_cls, new_node_id))

            for subcls in get_subclasses(cls):
                if subcls not in visited_classes:
                    queue.append((subcls, new_node_id))

    # Updated methods
    def get_ancestor_classes(self, node_id):
        ancestor_classes = []
        current_id = node_id
        while current_id is not None:
            node = self.conversation_tree.nodes.get(current_id)
            if not node:
                break
            ancestor_classes.append(node['name'])
            current_id = node.get('parent_id')
        return ancestor_classes[::-1]

    def validate_response(self, response):
        response_json = json.loads(response.strip())
        validated_response = LLMResponse.model_validate(response_json)
        return validated_response

    def start(self):
        root_class = self.ontology.ANNConfiguration
        if root_class is None:
            print(f"Class 'ANNConfiguration' does not exist in the ontology.")
            return

        visited_classes = set()
        self.handle_class(root_class, parent_id=0)


def main():

    # query_engine = LocalRagEngine().get_rag_query_engine()
    # response = query_engine.query("hello llama")
    # print(response)
    # return

    from utils.pdf_loader import load_pdf
    file_path = "data/hand_processed/AlexNet.pdf"  # Replace with your actual file path
    documents = load_pdf(file_path)
    # Combine all the page contents into a single string
    paper_content = "\n".join([doc.page_content for doc in documents])
    llm = OllamaLLMModel(temperature=0.5,top_k=7)

    onto = get_ontology(f"./data/owl/{C.ONTOLOGY.FILENAME}").load()
    # onto_prompts_json = load_ontology_questions("rag/tree_prompting/test_ontology_prompts.json")   


    tree = ConversationTree()
    questioner = OntologyTreeQuestioner(
        ontology=onto,
        conversation_tree=tree,
        # onto_prompts=onto_prompts_json,
        llm=llm,
        paper_content=paper_content
    )
    questioner.start()
    tree.save_to_json("output/conversation_tree.json")



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

def get_ollama_prompt() -> str:
    return """
You are provided with the following things:
1 - A question to answer delimited by triple backticks `.
2 - Previously, a paper as context written about a neural network architecture.
Your task is to perform the following actions: 
1 - Summarize the following question with details from the provided paper.
2 - Break down the question step-by-step to reach a clear conculsion ot the question.
3 - List each relevant ontology entity names the question asks for.
4 - Output a json object of ontology entity names that contains the following key: instance_names.

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

def generate_prompt(class_name, ancestor_classes, paper_content):
    prompt = f"""
You are an expert in ontology population. Based on the following information:

- **Class**: {class_name}
- **Ancestor Classes**: {', '.join(ancestor_classes) if ancestor_classes else 'None'}

Below is a research paper that provides detailed information about the class and its context:

\"\"\"
{paper_content}
\"\"\"

Please perform the following tasks:
1. Generate a question to instantiate a name for the class.
2. Relate the class to its ancestor classes where applicable.
"""
    return prompt

if __name__ == "__main__":
    main()
