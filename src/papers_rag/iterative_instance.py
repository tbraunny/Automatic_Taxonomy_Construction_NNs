from papers_rag.clean_response import clean_response
from utils.owl_utils import create_class, get_class_properties, get_property_range_type, get_instance_class_properties
from owlready2 import Thing

def is_atomic(prop_type):
    """Simple check if property type is atomic or a class based on given property type."""
    return prop_type == "atomic"

def get_architecture_name(llm):
    """
    Prompt the model to provide the architecture name for the ANNConfiguration class.

    Args:
        llm: Language model interface to query.

    Returns:
        str: The name of the architecture (e.g., "AlexNet", "ResNet50").
    """
    name_prompt = (
        f"Define the name of a specific architecture for the class 'ANNConfiguration' based on existing 
        instances and research paper context "
        f"Provide only the name of a specific architecture for the class 'ANNConfiguration'. "
        f"Examples could include 'AlexNet', 'ResNet50', or 'VGG16'. Only provide the name."
        # or?
        # f"Provide name for the main class in JSON format as a list, e.g., 
        # ['ANNConfiguration': 'ResNet50'] or ['ANNConfiguration': 'AlexNet']"
    )

    response = llm.aquery(name_prompt)
    architecture_name = clean_response(response, expected_type="atomic") # needs refinement for architecture name
    return architecture_name



def instantiate_architecture(ontology, architecture_name):
    """
    Create an instance of the architecture under the ANNConfiguration class.

    Args:
        ontology: The ontology where the class will be instantiated.
        architecture_name (str): The name of the architecture (e.g., "AlexNet").

    Returns:
        Thing: The instantiated architecture instance.
    """
    ann_config_class = create_class(ontology, "ANNConfiguration", base_class=Thing)
    architecture_instance = ann_config_class(name=architecture_name)
    print(f"Created architecture instance '{architecture_name}' under 'ANNConfiguration'.")
    return architecture_instance

def iterative_define_and_instantiate_properties(llm, ontology, architecture_instance):
    """
    Iteratively define and instantiate properties for the architecture instance.

    Args:
        llm: Language model interface to query.
        ontology: The ontology containing properties and class definitions.
        architecture_instance: The instance of the architecture.
    """

    ann_config_class = ontology.ANNConfiguration
    properties = get_instance_class_properties(ontology, ann_config_class)

    queue = [(architecture_instance, properties)]

    while queue:
        # define a current instances properties
        current_instance, current_properties = queue.pop(0)

        for prop in current_properties:
            prop_type = get_property_range_type(prop) # class or atomic? are there alternatives?

            if prop_type == "atomic":
                # Prompt for the atomic property value
                atomic_prompt = f"What is the value of '{prop.name}' for '{current_instance.name}'?"
                prop_response = llm.aquery(atomic_prompt)
                prop_value = clean_response(prop_response, expected_type="atomic") # need to handle cleaning? 
                setattr(current_instance, prop.name, prop_value)
            else:
                # Complex property: create instance and queue for further definition
                base_class = prop.range[0] # Assume single class in range
                complex_class_name = base_class.name
                complex_instance = create_class(ontology, complex_class_name, base_class)(name=f"{current_instance.name}_{complex_class_name}")
                setattr(current_instance, prop.name, complex_instance)

                complex_prompt = (
                    f"Provide a specific name or description for the '{prop.name}' associated with "
                    f"the '{current_instance.name}' instance of '{base_class.name}'."
                    "Examples could include layer names like 'ConvLayer1' or 'MaxPoolLayer'."
                    # could return this in json format to parse or something else??
                )

                complex_name_response = llm.aquery(complex_prompt)
                complex_instance_name = clean_response(complex_name_response, expected_type="atomic") # clean atomic response 

                # Create instance of the complex class using the LLM-provided name
                complex_instance = create_class(ontology, base_class.name, base_class=base_class)(
                    name=complex_instance_name
                )
                setattr(current_instance, prop.name, complex_instance)

                # Queue the new complex instance for further property definition
                complex_properties = get_class_properties(ontology, base_class)
                queue.append((complex_instance, complex_properties))

        print(f"Completed defining properties for '{current_instance.name}'.")

"""

ontology = get_ontology("http://example.org/ont").load()
llm_predictor = LLMModel(model_name="llama3.2:1b").get_llm()  # Instantiate your LLMModel with the desired model

iterative_instance_definition(llm_predictor, ontology, "ANNConfiguration")

ontology.save(file="output.owl")
print("Ontology saved.")

"""