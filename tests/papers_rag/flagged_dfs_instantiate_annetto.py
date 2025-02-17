from owlready2 import Ontology, ThingClass, Thing, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties,
    get_connected_classes,
    get_subclasses,
    create_cls_instance, assign_object_property_relationship
)
from utils.annetto_utils import (
    requires_final_instantiation,
    subclasses_requires_final_instantiation
)
from utils.llm_service import init_engine, query_llm

# Classes to omit from instantiation
OMIT_CLASSES = set(["DataCharacterization", "Regularization"])

# Special parent classes (where special rules apply)
PARENT_CLASSES = set([
    "LossFunction", "RegularizerFunction", "ActivationLayer", 
    "NonDiff", "Smooth", "AggregationLayer", "NoiseLayer"
])


def dfs_instantiate_annetto(ontology: Ontology):
    """
    Recursively instantiate an ANN ontology using branch-specific ancestor context.
    Each branch (e.g., a GAN with 'Generator' and 'Discriminator') is processed separately,
    and the LLM is provided with the appropriate context so that children (like Layer)
    know which branch (Generator or Discriminator) they belong to.
    """

    # Determines if a class should be instantiated
    def _needs_final_instantiation(cls: ThingClass) -> bool:
        return requires_final_instantiation(cls)

    # Determines if the subclasses of a class should be instantiated
    def _needs_subclass_instantiation(cls: ThingClass) -> bool:
        if cls.name in PARENT_CLASSES:
            return True
        # return subclasses_requires_final_instantiation(cls)
    
    # Instantiate a class instance
    def _instantiate_cls(cls:ThingClass, instance_name:str) -> Thing:
        instance = create_cls_instance(cls, instance_name)
        print(f"Instantiated {cls.name} with name: {instance_name}")
        return instance

    # Link a child instance to its parent instance via object property
    def _link_instances(parent_instance: Thing, child_instance: Thing):
        assign_object_property_relationship(ontology, parent_instance, child_instance)

    # Query the LLM
    def _query_llm(instructions:str, prompt:str) -> str:
        full_prompt = f"{instructions}\n{prompt}"
        try:
            print(f"LLM query: {full_prompt}")
            response = query_llm(full_prompt)
            print(f"LLM query response: {response}")
            return response.strip()
        except Exception as e:
            print(f"LLM query error: {e}")
            return ""

    # Build a prompt for a given class, including the branch (ancestor) context
    def _get_cls_prompt(cls: ThingClass, llm_context: list[str]) -> str:
        context_str = " > ".join(llm_context) if llm_context else "None"
        return (
            f"Current branch context: {context_str}.\n"
            f"Considering the neural network described in the paper, list the names of the relevant instances "
            f"for the class '{cls.name}'. If there are multiple, return a comma-separated list."
        )

    # Query the LLM for instance names for a class, using the current branch context
    def _get_cls_instances(cls: ThingClass, llm_context: list[str]) -> list[str]:
        prompt = _get_cls_prompt(cls, llm_context)
        instructions = (
            "Return a comma-separated list of instance names. "
            "For example: 'Convolutional Layer, Fully-Connected Layer'."
            f"\n\n{prompt}"
        )
        instance_names = _query_llm(instructions, prompt)
        print(type(instance_names))
        if not instance_names:
            return []
        # instance_names = [name.strip() for name in response.split(",") if name.strip()]
        print(f"LLM returned instances for {cls.name} with context {llm_context}: {instance_names}")
        return instance_names

    # Instantiate data properties for an instance by querying the LLM
    def _instantiate_data_property(cls: ThingClass, instance: Thing, llm_context: list[str]):
        data_props = get_class_data_properties(ontology, cls)
        for prop in data_props:
            if not hasattr(instance, prop.name) or not getattr(instance, prop.name):
                prompt = (
                    f"Current branch context: {' > '.join(llm_context)}.\n"
                    f"For the instance '{instance.name}' of class '{cls.name}', provide a concise value "
                    f"for the data property '{prop.name}'."
                )
                instructions = "Return a single value."
                value = _query_llm(instructions, prompt)
                if value:
                    try:
                        if prop.python_type in [int, float]:
                            converted_value = float(value) if prop.python_type == float else int(value)
                            setattr(instance, prop.name, converted_value)
                        else:
                            setattr(instance, prop.name, value)
                        print(f"Set data property '{prop.name}' of {instance.name} to {value}.")
                    except Exception as e:
                        print(f"Error converting value for {prop.name} on {instance.name}: {e}")
                        setattr(instance, prop.name, value)
                else:
                    print(f"No value returned for data property '{prop.name}' of {instance.name}.")

    # Recursion for _process_entity to handle connected classes and subclasses
    def _process_entity_recursion(cls, processed_classes, full_context: list[Thing],  llm_context: list[str]):
                # Process connected classes via object properties.
                connected_classes = get_connected_classes(cls, ontology)
                if connected_classes:
                    for conn_cls in connected_classes:
                        if isinstance(conn_cls, ThingClass):
                            _process_entity(conn_cls, processed_classes, full_context, llm_context)
                        else:
                            print(f"Encountered non-class connection from {cls.name}.")

                # Process subclasses if they need instantiation.
                subclasses = get_subclasses(cls)
                if subclasses:
                    for subclass in subclasses:
                        # if _needs_subclass_instantiation(subclass):
                            _process_entity(subclass, processed_classes, full_context, llm_context)
                        # else:
                        #     print(f"Skipping subclass instantiation for {subclass.name}.")

    # Main recursive function that processes a class and its relationships
    # full_context: list of all ancestor instances (Things) used for linking
    # llm_context: list of instance names (from final instantiations) used in LLM queries
    def _process_entity(cls: ThingClass, processed_classes: set, full_context: list[Thing],  llm_context: list[str]):
        if cls in processed_classes or cls.name in OMIT_CLASSES:
            return
        processed_classes.add(cls)

        if _needs_final_instantiation(cls):
            # Final instantiation: ask the LLM for instance names.
            instance_names = _get_cls_instances(cls, llm_context)
            if instance_names:
                for name in instance_names:
                    instance = _instantiate_cls(cls, name)
                    new_full_context = full_context + [instance]
                    new_llm_context = llm_context + [name]
                    _instantiate_data_property(cls, instance, new_llm_context)
                    if full_context:
                        parent_instance = full_context[-1]
                        _link_instances(parent_instance, instance)
                    _process_entity_recursion(cls, processed_classes, new_full_context, new_llm_context)
            else:
                # If no names were returned, continue recursion with the same contexts.
                print(f"No instances returned for {cls.name}.####################")
                _process_entity_recursion(cls, processed_classes, full_context, llm_context)
        else:
            # Generic instantiation: generate a generic name.
            # For example, if llm_context is ['convolutional_network', 'convolutional_layer'] and cls.name is 'AggregationLayer',
            # then generic_name becomes 'convolutional_network-convolutional_layer-aggregationlayer'
            generic_name = "-".join(llm_context + [cls.name.lower()]) if llm_context else cls.name.lower()
            instance = _instantiate_cls(cls, generic_name)
            new_full_context = full_context + [instance]
            # Do not add the generic name to llm_context.
            new_llm_context = llm_context
            _instantiate_data_property(cls, instance, new_llm_context)
            if full_context:
                parent_instance = full_context[-1]
                _link_instances(parent_instance, instance)
            _process_entity_recursion(cls, processed_classes, new_full_context, new_llm_context)



        # # Get instance names for the current class using the branch context.
        # instance_names = _get_cls_instances(cls, ancestor_context)

        # # Process the class instances.
        # if instance_names:
        #     for name in instance_names:
        #         instance = _instantiate_cls(cls, name)
        #         # Create a new branch context including this instance.
        #         new_context = ancestor_context.append(instance)

        #         # Process data properties for the instance.
        #         _instantiate_data_property(cls, instance, new_context)

        #         # Link the instance to its parent instance.
        #         if ancestor_context:
        #             parent_instance = ancestor_context[-1]
        #             _link_instances(parent_instance, instance)
        #         # Recurse on instance to process connected classes and subclasses.
        #         _process_entity_recursion(cls,processed_classes,new_context)
        # else:
        #     # No instances returned, so process connected classes and subclasses.
        #     _process_entity_recursion(cls,processed_classes,ancestor_context)

    # Check for the required root class.
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return

    # Initialize the LLM engine with the document context (e.g., the GAN paper).
    json_file_path = "data/alexnet/doc_alexnet.json"
    init_engine(json_file_path)

    processed_classes = set()

    processed_classes.add(ontology.ANNConfiguration)
    processed_classes.add(ontology.TrainingStrategy)


    # Create the root instance of the ANN.
    root_instance = create_cls_instance(ontology.ANNConfiguration, "ImageNet Classification with Deep Convolutional Neural Networks")

    full_context = [root_instance]
    llm_context = []  

    # # Start processing top-level classes with an empty branch context.
    # if hasattr(ontology, "Network"):
    #     _process_entity(ontology.Network, processed_classes, full_context, llm_context)

    # Will process Network class like this for now because describing the difference 
    # between Network like convolutional network and kinds of layers is difficult
    if hasattr(ontology, "Network"):
        network_instances = []
        network_instances.append(create_cls_instance(ontology.Network, "Convolutional Network"))

        for network_instance in network_instances:
            assign_object_property_relationship(ontology, root_instance, network_instance)

            for connected_class in get_connected_classes(ontology.Network, ontology):
                new_full_context = full_context + [connected_class]
                new_llm_context = llm_context + [connected_class.name]
                _process_entity(connected_class, processed_classes, new_full_context, new_llm_context)


    # TODO Skipping TrainingStrategy for now
    # if hasattr(ontology, "TrainingStrategy"):
    #     _process_entity(ontology.TrainingStrategy, processed_classes, full_context, llm_context)

    # TODO link the root instance to the top-level instances

    print("An ANN has been successfully instantiated.")


if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()

    with ontology:
        dfs_instantiate_annetto(ontology)

        new_file_path = "annett-o-test.owl"
        ontology.save(file=new_file_path, format="rdfxml")
        print(f"Ontology saved to {new_file_path}")
