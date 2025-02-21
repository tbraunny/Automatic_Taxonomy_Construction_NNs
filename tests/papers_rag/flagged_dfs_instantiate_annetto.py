from owlready2 import Ontology, ThingClass, Thing, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties, get_connected_classes, get_subclasses, 
    create_cls_instance, split_camel_case, create_cls_instance, assign_object_property_relationship
)
from utils.annetto_utils import requires_final_instantiation, subclasses_requires_final_instantiation
from utils.llm_service import init_engine, query_llm
from utils.json_utils import get_json_value

OMIT_CLASSES = ["DataCharacterization", "Regularization"]

PARENT_CLASSES = set(["Layer", "LossFunction", "RegularizerFunction", "ActivationLayer", "NonDiff", "Smooth", "AggregationLayer", "NoiseLayer", "TaskCharacterization", "RegularizerFunction", "LossFunction"])

def dfs_instantiate_annetto(ontology: Ontology):
    """
    
    """
    def _needs_instantiation(cls: ThingClass) -> bool:
        """Checks if a class should be instantiated."""
        return requires_final_instantiation(cls)
    
    def _needs_subclass_instantiation(cls: ThingClass) -> bool:
        """Checks if new subclasses should be explored."""
        return subclasses_requires_final_instantiation(cls)

    def _instantiate_data_property(cls:ThingClass, instance:Thing, ancestor_things: list[Thing]=None):
        """Write all data properties of a class, if any."""
        props = get_class_data_properties(ontology, cls)
        if props:
            for prop in props:
                # file.write(f"{indent}        - Data Prop: {prop.name} (atomic)\n")

                # Query llm if a value for this data property exists

                # Add it to instance
                pass

    def _instantiate_cls(cls:ThingClass, instance_name:str) -> Thing:
        instance = create_cls_instance(cls, instance_name)
        return instance

    def _query_llm(prompt)-> str:
        return query_llm(prompt)
    
    def _get_cls_prompt(cls:ThingClass, network_name:str) -> str:
        """Given a class, get an associated llm prompt for it to be instantiated"""

        # Get definition of the parent class
        class_definition = _get_cls_definition(cls)

        subclasses = get_subclasses(cls) # Gets list of cls subclasses
        subclass_names = [subclass.name for subclass in subclasses] # Get list of names of subclasses
        subclass_names = split_camel_case(subclass_names) # Split camel case names of subclasses
        subclass_names = ", ".join(subclass_names) # Join names of subclasses into a string

        prompt = (
            f"""List each instance of {class_definition} in the {network_name} sequentially. """
            # f"""Include repeated occurrences, exactly as they appear in the network structure."""
            f"""Do not generalize or collapse duplicate types; instead, explicitly enumerate each instance.\n\n"""
            f"""The network to be considered in the context is {network_name}. Do not consider context from another type of network.\n\n"""
            f"""Use the following examples and output format as a guide: {subclass_names}\n"""
            f"""For example, if the entity has multiple instances, they should be listed as """
            f"""Entity 1, Entity 2, etc.\n """
            f"""Ensure the output maintains the correct order of entities as found in the context."""
        )
        return prompt

    def _get_cls_instances(cls:ThingClass) -> Thing:
        # Get prompt for given class

        # combine prompt with RAG context

        # combine prompt with general llm instructions 

        # Query LLM on prompt

        # Validate llm response format (i.e. pydantic for json)

        # Parse prompt into list of instance names

        # list of instance objects = _instantiate_cls (instance_names)

        # return list of instance objects
        return ["Convolutional Layer", "Fully-Connected Layer", "Attention Layer"]
    
    def _get_cls_definition(cls, json_prompts_path:str="tests/papers_rag/class_definitions.json") -> str:
        return get_json_value(cls.name, json_prompts_path)
        
    def _get_subclasses_instances(cls:ThingClass, network_name:str) -> list[Thing]:
        """ Returns a list of instances of the subclasses of a class"""

        # combine prompt with RAG context
        prompt = _get_cls_prompt(cls, network_name,network_name)

        # Query LLM on prompt
        named_instances = _query_llm(prompt)

        print("\n\nNamed Instances", named_instances)####################

        # Create list of instance objects
        instances = []
        for instance in named_instances:
            instance_thing = _instantiate_cls(cls, instance)
            print(f"Instance: {instance_thing}")####################
            instances.append(instance_thing)

        # return list of instance objects
        return instances

    

    def _process_entity(cls: ThingClass, label: str, processed_classes: set, ancestor_things: list[Thing] = None):
        """
        Process an entity (class, connected class, or subclass)
        """

        # Need logic to handle instantiating a class not in 'parent classes' list but also has no subclasses 

        if ancestor_things is None:
            ancestor_things = []

        # Skip if already processed, preventing loops
        # Skip if in omit list
        if cls in processed_classes or cls.name in OMIT_CLASSES:
            return

        # Add class to processed set
        processed_classes.add(cls)

        # ASSUMPTION: Process instantiations by premarked parent classes 
        if cls.name in PARENT_CLASSES:
            instances = _get_subclasses_instances(cls, ancestor_things[0].name)

            # Connect instance to parent instance by object property
            for instance in instances if instances else []:
                assign_object_property_relationship(ontology, ancestor_things[-1], instance)
        
        # ASSUMPTION: If using simplified Processing of Layer Class 
        if cls.name == "Layer":
            return

        # Fuck, need logic for if instances exist and dont exist
        if instances:
            for instance in instances:

                # Append current instance to ancestor chain.
                new_ancestor_things = ancestor_things + [instance]
            
                # Process data properties into instance object (probably?).
                _instantiate_data_property(cls, instance, new_ancestor_things)

                def recurse():
                    # Process connected classes via object properties.
                    connected_classes = get_connected_classes(cls, ontology)
                    if connected_classes:
                        for connected_class in connected_classes:
                            if isinstance(connected_class, ThingClass):
                                _process_entity(connected_class, "Connected Class", processed_classes, ancestor_things)
                            else:
                                print(f"################ Non-Class Connection? ################\n")

                    # Process subclasses.
                    subclasses = get_subclasses(cls)
                    if subclasses:
                        for subclass in subclasses:
                            _process_entity(subclass, "Subclass", processed_classes, ancestor_things)
        else:
            return

    # Check for the required key class.
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return
    
    paper_json_doc_file_path = "data/alexnet/doc_alexnet.json"
    init_engine(paper_json_doc_file_path) # Initialize LLM engine

    processed_classes = set()

    # Instantiate the root class 
    # ASSUMPTION: Assume initial network is 'Alexnet'
    root_instance = create_cls_instance(ontology.ANNConfiguration, "Alexnet")

    # Process the top-level classes

    # ASSUMPTION: Assume initial network is 'Convolutional Network'
    if hasattr(ontology, "Network"):
        network_instances = []
        network_instances.append(create_cls_instance(ontology.Network, "Convolutional Network"))

        for network_instance in network_instances:
            assign_object_property_relationship(ontology, root_instance, network_instance)

            for connected_class in get_connected_classes(ontology.Network, ontology):
                _process_entity(connected_class, "Connected Class", processed_classes, [network_instance])

    # if hasattr(ontology, "TrainingStrategy"):
    #     _process_entity(ontology.TrainingStrategy, "Class", processed_classes)

    # Need to have instances of TrainingStrategy be in the range of object hasTrainingStrategy and root_instance be in the domain

    print("ANN has been instantiated.")

if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()    

    with ontology:
        dfs_instantiate_annetto(ontology)

        new_file_path = "annett-o-0.test.owl"
        ontology.save(file=new_file_path, format="rdfxml")
        print(f"Ontology saved to {new_file_path}")
