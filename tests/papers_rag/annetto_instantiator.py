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

import logging
from scripts.write_annetto_structure import write_ontology_structure_to_file

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(filename='annetto_instantiator.log', encoding='utf-8', level=logging.INFO, force=True)

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

LAYER_EXAMPLES = [ "GRULayer",
                    "LSTMLayer",
                    "ConvolutionLayer",
                    "SeparableConvolutionLayer",
                    "DeconvolutionLayer",
                    "SeparableDeconvolutionLayer",
                    "FullyConnectedLayer",
                    "BidirectionalRNNLayer",
                    "ConcatLayer",
                    "MultiplyLayer",
                    "PoolingLayer",
                    "SumLayer",
                    "UpscaleLayer",
                    "NoiseLayer",
                    "DropoutLayer",
                    "BatchNormLayer",
                    "FlattenLayer",
                    "CloneLayer",
                    "SplitLayer" ]

TASK_CHARACTERIZATION_EXAMPLES = [
    "Classification",
    "Regression",
    "Segmentation",
    "Object Detection",
    "Instance Segmentation",
    "Semantic Segmentation",
    "Pose Estimation",
    "Image Generation",
    "Image Translation",
    "Image Restoration",
    "Image Denoising",
    "Image Colorization",
]

class AnnettoInstantiator():
    def __init__(self, ontology_path, output_ontology_path, json_context_doc_path):
        self.ontology = get_ontology(ontology_path).load()
        self.output_ontology_path = output_ontology_path
        self.engine = init_engine(json_context_doc_path)
        self.instantiated_classes = set()
        self.instantiated_instances = set()
        self.instantiated_subclasses = set()

    def save_ontology(self):
        self.ontology.save(file=self.output_ontology_path, format="rdfxml")
        logger.info(f"Ontology saved to {self.output_ontology_path}.")

    def populate_ontology(self):
        if not hasattr(self.ontology, 'ANNConfiguration'):
            logger.info("Error: Class 'ANNConfiguration' not found in ontology.")
            return
    
        visited_classes = set()    

        # Instantiate the root ANNConfiguration.
        self.root_instance = self.instantiate_class(self.ontology.ANNConfiguration, "ImageNet") # TODO: Change this an llm defined name
        visited_classes.add(self.ontology.ANNConfiguration)
        
        parent_instantiation_path = [self.root_instance]
        llm_context = []

        if hasattr(self.ontology, "Network"):
            # Get the object property from ANNConfiguration to Network
            logger.info(f"Processing Network class using{self.ontology.ANNConfiguration, self.ontology.Network}")
            object_property = get_object_properties_with_domain_and_range(self.ontology, self.ontology.ANNConfiguration, self.ontology.Network)
            
            # Ask LLM for a comma-separated list of Networks
            network_names = self.get_network_class_instances()

            if network_names:
                for net_name in network_names:
                    logger.info(f"Processing network {net_name}...")
                    net_instance = self.instantiate_class(self.ontology.Network, net_name)
                    self.link_instances(self.root_instance, net_instance, object_property)
                    
                    updated_path = parent_instantiation_path + [net_instance]
                    new_llm_context = llm_context + [net_instance.name]
                    
                    # Process any classes connected to "Network"
                    for conn in get_connected_classes(self.ontology.Network, self.ontology):
                        if isinstance(conn, ThingClass):
                            self.process_entity(conn, visited_classes, updated_path, new_llm_context)
            else:
                logger.error("No networks found in LLM context.")
        if hasattr(self.ontology, "TaskCharacterization"):
            # Get the object property from ANNConfiguration to Network
            logger.info(f"Processing TaskCharacterization class using{self.ontology.ANNConfiguration, self.ontology.TaskCharacterization}")
            object_property = get_object_properties_with_domain_and_range(self.ontology, self.ontology.ANNConfiguration, self.ontology.TaskCharacterization)
            
            # Ask LLM for a single TaskCharacterization
            task_characterization = self.get_network_class_instances()

            if type(task_characterization) is str:
                task_characterization = task_characterization.split(", ")
                if len(task_characterization) > 1:
                    logger.error("More than one task characterization found in LLM context. Only one task characterization is allowed.")
                    return
            
            if task_characterization:
                for task_name in task_characterization:
                    logger.info(f"Processing task characterization {task_name}...")
                    task_instance = self.instantiate_class(self.ontology.TaskCharacterization, task_name)
                    self.link_instances(self.root_instance, task_instance, object_property)
                    
                    updated_path = parent_instantiation_path + [task_instance]
                    new_llm_context = llm_context + [task_instance.name]
                    
                    # Process any classes connected to "TaskCharacterization"
                    for conn in get_connected_classes(self.ontology.TaskCharacterization, self.ontology):
                        if isinstance(conn, ThingClass):
                            self.process_entity(conn, visited_classes, updated_path, new_llm_context)
            else:
                logger.error("No Task Characterizations found in LLM context.")

    def _query_llm(self, instructions: str, prompt: str) -> str:
        full_prompt = f"{instructions}\n{prompt}"
        try:
            print(f"LLM query: {full_prompt}")
            response = query_llm(full_prompt)
            print(f"LLM query response: {response}")
            return response
        except Exception as e:
            print(f"LLM query error: {e}")
            return ""
        
    def instantiate_class(self, cls: ThingClass, instance_name: str) -> Thing:
        instance = create_cls_instance(cls, instance_name)
        logger.info(f"Instantiated {cls.name} with name: {instance_name}")
        return instance

    def link_instances(self, parent_instance: Thing, child_instance: Thing, object_property):
        """
        Assign the given object property relationship between parent and child.
        """
        if object_property is None:
            print(f"Warning: No object property provided for linking {parent_instance} and {child_instance}.")
            return
        logger.info(f"Linking {parent_instance.name} and {child_instance.name} via {object_property}...")
        print(f"Linking {parent_instance.name} and {child_instance.name} via {object_property}...")
        assign_object_property_relationship(parent_instance, child_instance, object_property)

    def build_generic_class_prompt(self, cls: ThingClass, llm_context: list[str]) -> str:
        context_str = " > ".join(llm_context) if llm_context else "None"
        return (f"Current branch context: {context_str}.\n"
                f"Considering the neural network described in the paper, list the names of the relevant instances "
                f"for the class '{cls.name}'. If there are multiple, return a comma-separated list.")

    def build_network_class_prompt(self) -> str:
        # TODO: FIX THIS. Currently special case for network
        return (
            "Based on the neural network architecture described in the paper, identify all sub-networks or component "
            "branches that make up the full model. A single model can have multiple branches or networks (e.g., a GAN "
            "with a generator and discriminator, or a multi-branch convolutional model with separate paths). "
            "If multiple branches exist, return them as a comma-separated list (for example: "
            "'Generator Network, Discriminator Network' or 'Convolutional Branch A, Convolutional Branch B'). If there is only one major network, "
            "Simply provide that single name. "
        )
    
    def get_network_class_instances(self) -> list[str]:
        prompt = self.build_network_class_prompt()
        instructions = "Return a comma-separated list of network names. For example: 'Generator Network, Discriminator Network'.\n\n" + prompt
        response = self._query_llm(instructions, prompt)
        if not response:
            return []
        network_names = response

        if type(network_names) is str:
            network_names = network_names.split(", ")

        logger.info(f"LLM returned network names: {network_names}")
        print(f"LLM returned network names: {network_names} type {type(network_names)}")
        return network_names

    def get_class_instances_from_llm(self, cls: ThingClass, llm_context: list[str]) -> list[str]:
        prompt = self.build_generic_class_prompt(cls, llm_context)
        instructions = ("Return a comma-separated list of instance names. "
                        "For example: 'Convolutional Layer, Fully-Connected Layer'.\n\n" + prompt)
        response = self._query_llm(instructions, prompt)
        if not response:
            return []
        # Here we assume the LLM returns a comma-separated list.
        instance_names = response
        
        print(f"LLM returned instances for {cls.name} with context {llm_context}: {instance_names}")
        return instance_names
    
    def process_connected_class(self, cls: ThingClass, visited_classes: set, parent_instantiation_path: list[Thing], llm_context: list[str]):
        """
        Process a connected class by instantiating it and its subclasses, and linking it to the parent class.
        """
        if cls in visited_classes:
            logger.info(f"Skipping {cls.name} as it has already been visited.")
            return
        elif cls.name in OMIT_CLASSES:
            logger.info(f"Skipping {cls.name} as it is in the omit list.")
            return
        
        visited_classes.add(cls)

        if cls.name == self.ontology.Layer.name:
            self.process_layer_classes(cls, visited_classes, parent_instantiation_path, llm_context)
        
    #     # Get the object property from the parent class to the connected class.
    #     object_property = get_object_properties_with_domain_and_range(self.ontology, parent_instantiation_path[-1].__class__, cls)
        
    #     # Instantiate the connected class.
    #     instance_names = self.get_class_instances_from_llm(cls, llm_context)
    #     if not instance_names:
    #         return
        
    #     for instance_name in instance_names:
    #         instance = self.instantiate_class(cls, instance_name)
    #         self.link_instances(parent_instantiation_path[-1], instance, object_property)
            
    #         updated_path = parent_instantiation_path + [instance]
    #         new_llm_context = llm_context + [instance_name]
            
    #         # Process any subclasses of the connected class.
    #         for subclass in get_subclasses(cls, self.ontology):
    #             self.process_entity(subclass, visited_classes, updated_path, new_llm_context, is_subclass=True)

    def process_layer_classes(self, cls: ThingClass, visited_classes: set, parent_instantiation_path: list[Thing], llm_context: list[str]):
        """
        Process layer classes by instantiating them and their subclasses, and linking them to the parent class.
        """

        # TODO: remove this check?
        # if cls in visited_classes:
        #     logger.info(f"Skipping {cls.name} as it has already been visited.")
        #     return
        # elif cls.name in OMIT_CLASSES:
        #     logger.info(f"Skipping {cls.name} as it is in the omit list.")
        #     return
                
        visited_classes.add(cls)

        # TODO: Update layer examples to be more robust 
        prompt = (
            f"Current branch context: {' > '.join(llm_context) if llm_context else 'None'}.\n"
            "The neural network described in the paper may contain multiple layers of various types. "
            "Please list the types of all layers you find, along with how many of each type there are, "
            "in the format '<count> <layer type>'.\n\n"
            "For example: '7 Attention Layer, 2 RNN Layer'.\n"
            "If there are multiple instances of the same layer, please include them with a count.\n"
            f"Examples of layers include: {", ".join(LAYER_EXAMPLES)}. "
            "If you encounter any layer types not in this list, please include them as well.\n"
            "Return the result as a comma-separated list.\n"
        )
        instructions = "Return the count of unique layers in the provided paper as a comma-separated list."

        response = self._query_llm(instructions, prompt)

        if not response:
            logger.info("No response from LLM for layer classes.")
            return
        
        layer_instances = response
        if type(layer_instances) is str:
            layer_instances = layer_instances.split(", ")

        for layer_name in layer_instances:
            instance = self.instantiate_class(cls, layer_name)
            updated_path = parent_instantiation_path + [instance]
            new_llm_context = llm_context + [layer_name]

            if parent_instantiation_path:
                instance = self.instantiate_class(cls, layer_name)
                linking_property = get_object_properties_with_domain_and_range(self.ontology, parent_instantiation_path[-1].__class__, cls)
                self.link_instances(parent_instantiation_path[-1], instance, linking_property)


        # Get the object property from the parent class to the connected class.
        print(f"TEST FINDING OBJ PROP FOR LAYER with {parent_instantiation_path[-1].__class__}")
    
    def process_task_characterization(self, cls: ThingClass, visited_classes: set, parent_instantiation_path: list[Thing], llm_context: list[str]):
        visited_classes.add(cls)

        # TODO: Update layer examples to be more robust 
        prompt = (
            f"Current branch context: {' > '.join(llm_context) if llm_context else 'None'}.\n"
            f"The neural network architecture corresponding to the {parent_instantiation_path[-1].name} subnetwork described in the paper. "
            "has a specific task characterization. "
            "Please identify the type task characterization you find.\n\n"
            f"Examples of Task Characterizations include: {", ".join(TASK_CHARACTERIZATION_EXAMPLES)} etc. "
            "If you encounter a task not in the list, please do not hesitate and provide the name. "
        )
        instructions = "Return the task characterization of the provided subnetwork in the provided paper."

        response = self._query_llm(instructions, prompt)

        if not response:
            logger.info("No response from LLM for layer classes.")
            return
        
        layer_instances = response
        if type(layer_instances) is str:
            layer_instances = layer_instances.split(", ")

        for layer_name in layer_instances:
            instance = self.instantiate_class(cls, layer_name)
            updated_path = parent_instantiation_path + [instance]
            new_llm_context = llm_context + [layer_name]

            if parent_instantiation_path:
                instance = self.instantiate_class(cls, layer_name)
                linking_property = get_object_properties_with_domain_and_range(self.ontology, parent_instantiation_path[-1].__class__, cls)
                self.link_instances(parent_instantiation_path[-1], instance, linking_property)

    def process_entity(self, cls: ThingClass, visited_classes: set, parent_instantiation_path: list[Thing], llm_context: list[str]):
        if cls in visited_classes or cls.name in OMIT_CLASSES:
            logger.info(f"Skipping {cls.name} as it has already been visited.")
            return
        
        match cls.name:
            case "Layer":
                self.process_layer_classes(cls, visited_classes, parent_instantiation_path, llm_context)
            case "TaskCharacterization":
                self.process_task_characterization(cls, visited_classes, parent_instantiation_path, llm_context)
            case _:
                logger.info(f"Processing connected class {cls.name}...")
        

if __name__ == "__main__":
    logger.info("Starting Annetto Instantiator...")

    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()
    
    instantiator = AnnettoInstantiator(ontology_path, "annetto-test-output.owl", "data/alexnet/doc_alexnet.json")
    print(f"instantiator.ontology.TaskCharacterization {instantiator.ontology.TaskCharacterization}")
    instantiator.instantiate_class(instantiator.ontology.TaskCharacterization, "Classification")

    write_ontology_structure_to_file(instantiator.ontology, './annetto_structure_test.txt')

    instantiator.populate_ontology()
    # instantiator.save_ontology