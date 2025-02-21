from owlready2 import Ontology, ThingClass, Thing, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties,
    get_connected_classes,
    get_subclasses,
    create_cls_instance, 
    assign_object_property_relationship, 
    create_subclass,
    get_object_properties_with_domain_and_range, 
    split_camel_case
)
# from utils.annetto_utils import ()
from utils.llm_service import init_engine, query_llm

# Import fuzzy matching library.
from fuzzywuzzy import fuzz

# Classes to omit from instantiation.
OMIT_CLASSES = set(["DataCharacterization", "Regularization"])

# Organizational nodes – if such a class has subclasses then we want to perform organizational instantiation.
ORGANIZATIONAL_CLASSES = set([
    "LossFunction", "RegularizerFunction", "AggregationLayer", 
    "ModificationLayer", "SeparationLayer", "ObjectiveFunction", 
    "TaskCharacterization", 
])

OBJECTIVE_FUNCTION_CLASSES = set(["ObjectiveFunction", "LossFunction", "RegularizerFunction", "CostFunction"])
LAYER_CLASSES = set(["HiddenLayer", "InputLayer", "OutputLayer", "InOutLayer", "ActivationLayer", "ActivationFunction","NonDiff","Linear", "Smooth","AggregationLayer","ModificationLayer"])


def dfs_instantiate_annetto(ontology: Ontology):

    # def get_instantiation_type(cls: ThingClass) -> str:
    #     """
    #     Returns one of:
    #       - "organizational": if the class has subclasses and is in ORGANIZATIONAL_CLASSES.
    #       - "meaningful": if requires_final_instantiation(cls) is True (and not organizational).
    #       - "generic": otherwise (for connected classes without subclasses).
    #     """
    #     if get_subclasses(cls):
    #         if cls.name in ORGANIZATIONAL_CLASSES:
    #             return "organizational"
    #         else:
    #             if requires_final_instantiation(cls):
    #                 return "meaningful"
    #             else:
    #                 return "generic"
    #     else:
    #         if requires_final_instantiation(cls):
    #             return "meaningful"
    #         else:
    #             return "generic"

    def _instantiate_cls(cls: ThingClass, instance_name: str) -> Thing:
        instance = create_cls_instance(cls, instance_name)
        print(f"Instantiated {cls.name} with name: {instance_name}")
        return instance

    def _link_instances(parent_instance: Thing, child_instance: Thing, object_property):
        """
        Assign the given object property relationship between parent and child.
        """
        if object_property is None:
            print(f"Warning: No object property provided for linking {parent_instance} and {child_instance}.")
            return
        print(f"Linking {parent_instance.name} and {child_instance.name} via {object_property}...")
        assign_object_property_relationship(parent_instance, child_instance, object_property)


    def _query_llm(instructions: str, prompt: str) -> str:
        full_prompt = f"{instructions}\n{prompt}"
        try:
            print(f"LLM query: {full_prompt}")
            response = query_llm(full_prompt)
            print(f"LLM query response: {response}")
            return response
        except Exception as e:
            print(f"LLM query error: {e}")
            return ""
    
    def _get_cls_prompt(cls: ThingClass, llm_context: list[str]) -> str:
        context_str = " > ".join(llm_context) if llm_context else "None"
        return (f"Current branch context: {context_str}.\n"
                f"Considering the neural network described in the paper, list the names of the relevant instances "
                f"for the class '{cls.name}'. If there are multiple, return a comma-separated list.")
    
    def _get_cls_instances(cls: ThingClass, llm_context: list[str]) -> list[str]:
        prompt = _get_cls_prompt(cls, llm_context)
        instructions = ("Return a comma-separated list of instance names. "
                        "For example: 'Convolutional Layer, Fully-Connected Layer'.\n\n" + prompt)
        response = _query_llm(instructions, prompt)
        if not response:
            return []
        # Here we assume the LLM returns a comma-separated list.
        instance_names = response
        print(f"LLM returned instances for {cls.name} with context {llm_context}: {instance_names}")
        return instance_names
    
    def _instantiate_data_properties(cls: ThingClass, instance: Thing, llm_context: list[str]):
        data_props = get_class_data_properties(ontology, cls)
        if data_props:
            for prop in data_props:
                if not hasattr(instance, prop.name) or not getattr(instance, prop.name):
                    prompt = (f"Current branch context: {' > '.join(llm_context)}.\n"
                              f"For the instance '{instance.name}' of class '{cls.name}', provide a concise value "
                              f"for the data property '{prop.name}'.")
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
    
    def _process_connected_classes(cls: ThingClass, processed: set, full_context: list[Thing], llm_context: list[str]):
        connected = get_connected_classes(cls, ontology)
        if connected:
            for conn in connected:
                if isinstance(conn, ThingClass):
                    _process_entity(conn, processed, full_context, llm_context, is_subclass=False)
                else:
                    print(f"Encountered non-class connection from {cls.name}.")
    
    def _process_subclasses(cls: ThingClass, processed: set, full_context: list[Thing], llm_context: list[str]):
        subs = get_subclasses(cls)
        if subs:
            for sub in subs:
                _process_entity(sub, processed, full_context, llm_context, is_subclass=True)

    def _get_list_layer_classes(cls: ThingClass, processed: set = None, layer_class_names: set = None):
        if processed is None:
            processed = set()
        if layer_class_names is None:
            layer_class_names = set()

        # Avoid processing the same class multiple times
        if cls in processed:
            return layer_class_names

        processed.add(cls)

        # Iterate through direct subclasses
        for sub in cls.subclasses():
            # Only add if not in exclude_list
            if sub.name not in LAYER_CLASSES:
                layer_class_names.add(sub.name)
            # Recursive call
            _get_list_layer_classes(sub, processed, layer_class_names)

        return layer_class_names

    def _process_layer_classes(cls: ThingClass, processed: set, full_context: list[Thing], llm_context: list[str], layer_class_names: set[str]):
        layer_class_names = set()

        layer_class_names = _get_list_layer_classes(cls, layer_class_names)

        layer_class_names = split_camel_case(layer_class_names)

        prompt = (f"Current branch context: {' > '.join(llm_context) if llm_context else 'None'}.\n"
                    f"Name and count each instance of layers in the neural network described in the paper. \n"
                    "For example: '7 Attention Layer, 2 RNN Layer'. "
                    # "If there are multiple instances of the same layer, list them sequentially, incuding repeats. \n"
                    f"Examples of layers include: {layer_class_names}. If you encounter a layer not in the list, please do not hesitate and provide the name.")
        instructions = ""

        response = _query_llm(instructions, prompt)
        if not response:
            print(f"No response for layer classes.")
            return
        layer_instances = response

        for name in layer_instances:
            instance = _instantiate_cls(cls, name)
            new_context = full_context + [instance]
            new_llm_context = llm_context + [name]
            # _instantiate_data_properties(cls, instance, new_llm_context)
            if full_context:
                parent_instance = full_context[-1]
                _link_instances(parent_instance, instance, object_property)

    
    def _process_entity(cls: ThingClass, processed: set, full_context: list[Thing], 
                        llm_context: list[str], is_subclass: bool = False, object_property=None):
        if cls in processed or cls.name in OMIT_CLASSES:
            return
        processed.add(cls)

        if cls is ontology.Layer:
            _process_layer_classes(cls, processed, full_context, llm_context, object_property)

        return
        
        inst_type = get_instantiation_type(cls)
        print(f"Processing {cls.name}: instantiation type = {inst_type}, is_subclass = {is_subclass}")
        
        # Helper to update branch context: when refining a parent instance with a subclass,
        # we “replace” the last element of full_context.
        def update_context(instance: Thing):
            if is_subclass and full_context:
                return full_context[:-1] + [instance]
            else:
                return full_context + [instance]
        
        if inst_type == "organizational":
            known_subclasses = get_subclasses(cls)
            known_map = {subcls.name.lower(): subcls for subcls in known_subclasses}
            known_names_str = ", ".join(known_map.keys())
            prompt = (f"Current branch context: {' > '.join(llm_context) if llm_context else 'None'}.\n"
                      f"For the organizational class '{cls.name}', the known subclasses are: {known_names_str}.\n"
                      "Based on the paper, list the relevant instance types for this category. "
                      "Include instance names that match the known subclasses as well as any new types not in the list. "
                      "Return a comma-separated list.")
            instructions = ""
            response = _query_llm(instructions, prompt)
            if not response:
                print(f"No response for organizational instantiation of {cls.name}.")
                return
            candidate_instances = [name.strip() for name in response.split(",") if name.strip()]
            for candidate in candidate_instances:
                best_match = None
                best_score = 0
                for known_name in known_map.keys():
                    score = fuzz.ratio(candidate.lower(), known_name)
                    if score > best_score:
                        best_score = score
                        best_match = known_name
                threshold = 70  # adjust threshold as needed
                if best_match and best_score >= threshold:
                    chosen_subclass = known_map[best_match]
                    print(f"Candidate '{candidate}' matched known subclass '{chosen_subclass.name}' with score {best_score}.")
                else:
                    new_subclass_name = candidate.replace(" ", "")
                    print(f"Creating new subclass '{new_subclass_name}' under {cls.name} for candidate '{candidate}' (score: {best_score}).")
                    with ontology:
                        new_sub = create_subclass(ontology, new_subclass_name, cls)
                    chosen_subclass = new_sub
                    known_map[new_subclass_name.lower()] = new_sub
                instance = _instantiate_cls(chosen_subclass, candidate)
                new_full_context = full_context + [instance]
                new_llm_context = llm_context + [candidate]
                _instantiate_data_properties(chosen_subclass, instance, new_llm_context)
                if full_context:
                    parent_instance = full_context[-1]
                    _link_instances(parent_instance, instance, object_property)
                # Optionally process connected classes and subclasses:
                # _process_connected_classes(chosen_subclass, processed, new_full_context, new_llm_context, object_property)
                # _process_subclasses(chosen_subclass, processed, new_full_context, new_llm_context, object_property)
        
        elif inst_type == "meaningful":
            instance_names = _get_cls_instances(cls, llm_context)
            if not instance_names:
                print(f"No LLM response for meaningful instantiation of {cls.name}; aborting branch.")
                return
            for name in instance_names:
                instance = _instantiate_cls(cls, name)
                new_context = update_context(instance)
                new_llm_context = llm_context + [name]
                _instantiate_data_properties(cls, instance, new_llm_context)
                if full_context:
                    parent_instance = full_context[-1]
                    _link_instances(parent_instance, instance, object_property)
                _process_connected_classes(cls, processed, new_context, new_llm_context, object_property)
                _process_subclasses(cls, processed, new_context, new_llm_context, object_property)
        
        elif inst_type == "generic":
            generic_name = "-".join(llm_context + [cls.name.lower()]) if llm_context else cls.name.lower()
            instance = _instantiate_cls(cls, generic_name)
            new_context = update_context(instance)
            new_llm_context = llm_context  # do not update llm context for generic instantiation
            _instantiate_data_properties(cls, instance, new_llm_context)
            if full_context:
                parent_instance = full_context[-1]
                _link_instances(parent_instance, instance, object_property)
            _process_connected_classes(cls, processed, new_context, new_llm_context, object_property)
            _process_subclasses(cls, processed, new_context, new_llm_context, object_property)
    
    
    # Verify required root exists.
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return
    
    # Initialize the LLM engine with the document context.
    json_file_path = "data/alexnet/doc_alexnet.json"
    init_engine(json_file_path)
    
    processed = set()
    processed.add(ontology.ANNConfiguration)
    processed.add(ontology.TrainingStrategy)
    
    # Create the root instance.
    root_instance = _instantiate_cls(ontology.ANNConfiguration, "ImageNet Classification with Deep Convolutional Neural Networks")
    full_context = [root_instance]
    llm_context = []
    
    # Process the top-level Network class.
    if hasattr(ontology, "Network"):
        network_instance = _instantiate_cls(ontology.Network, "Convolutional Network")
        object_property = get_object_properties_with_domain_and_range(ontology, ontology.ANNConfiguration, ontology.Network)


        _link_instances(root_instance, network_instance, object_property)
        new_context = full_context + [network_instance]
        new_llm_context = llm_context + [network_instance.name]
        for conn in get_connected_classes(ontology.Network, ontology):
            if isinstance(conn, ThingClass):
                _process_entity(conn, processed, new_context, new_llm_context, is_subclass=False)
    
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
