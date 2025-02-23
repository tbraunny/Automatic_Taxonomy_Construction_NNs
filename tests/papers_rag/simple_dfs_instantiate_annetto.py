from owlready2 import Ontology, ThingClass, Thing, ObjectProperty, get_ontology
from utils.constants import Constants as C
from utils.owl_utils import (
    get_class_data_properties,
    get_connected_classes,
    get_subclasses,
    create_cls_instance, 
    assign_object_property_relationship, 
    create_subclass,
)
from utils.annetto_utils import (int_to_ordinal, split_camel_case)
from utils.llm_service import init_engine, query_llm

from fuzzywuzzy import fuzz

# Classes to omit from instantiation.
OMIT_CLASSES = set(["DataCharacterization", "Regularization"])

def dfs_instantiate_annetto(ontology: Ontology):

    def _instantiate_cls(cls: ThingClass, instance_name: str) -> Thing:
        instance = create_cls_instance(cls, instance_name)
        print(f"Instantiated {cls.name} with name: {instance_name}")
        return instance
    
    def _instantiate_generic_class(cls: ThingClass, full_context: list[Thing]):
        # Extract names from full_context and handle empty list properly
        context_names = [thing.name for thing in full_context] if full_context else []
        # Build the generic name
        generic_name = "-".join(context_names + [cls.name.lower()])
        # Instantiate the class
        instance = _instantiate_cls(cls, generic_name)
        
        return instance

    def _link_instances(parent_instance: Thing, child_instance: Thing, object_property:ObjectProperty):
        """
        Assign the given object property relationship between parent and child.
        """
        assign_object_property_relationship(parent_instance, child_instance, object_property)
        print(f"Linked {parent_instance.name} and {child_instance.name} via {object_property}.")

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
    
    def _recurse_connected_classes(cls: ThingClass, processed: set, full_context: list[Thing], llm_context: list[str]):
        connected = get_connected_classes(cls, ontology)
        if connected:
            for conn in connected:
                if isinstance(conn, ThingClass):
                    _process_entity(conn, processed, full_context, llm_context)
                else:
                    print(f"Encountered non-class connection from {cls.name}.")
    
    def _recurse_subclasses(cls: ThingClass, processed: set, full_context: list[Thing], llm_context: list[str]):
        subs = get_subclasses(cls)
        if subs:
            for sub in subs:
                _process_entity(sub, processed, full_context, llm_context)

    def _find_ancestor_network_instance(full_context: list[Thing]):
        """Find the network instance in the full context"""
        return next((thing for thing in full_context if ontology.Network in thing.is_a), None)
    
    def _process_objective_functions(full_context: list[Thing]):
        # Get the network instance from the full context.
        network_thing = _find_ancestor_network_instance(full_context)
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        
        network_thing_name = network_thing.name

        # Process the loss functions used by the network.
        loss_function_prompt = (
            f"Extract only the names of the loss functions used for the {network_thing_name}'s architecture and return the result in JSON format with the key 'answer'. "
            "Follow the examples below.\n\n"
            "Examples:\n"
            "Network: Discriminator\n"
            '{"answer": ["Binary Cross-Entropy Loss"]}\n'
            "Network: Discriminator\n"
            '{"answer": ["Wasserstein Loss", "Hinge Loss"]}\n\n'
            f"Now, for the following network:\nNetwork: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        loss_function_response = _query_llm("", loss_function_prompt)
        if not loss_function_response:
            print("No response for loss function classes.")
            return
        loss_function_names = loss_function_response  # expected to be a list of loss function names

        # For each loss function, determine its optimization direction and process its regularizers.
        for loss_name in loss_function_names:
            # Query whether this loss function is designed to minimize or maximize.
            loss_objective_prompt = (
                f"Is the {loss_name} function designed to minimize or maximize its objective function? "
                "Please respond with either 'minimize' or 'maximize' in JSON format using the key 'answer'.\n\n"
                "**Clarification:**\n"
                "- If the function is set up to minimize (e.g., cross-entropy, MSE), respond with 'minimize'.\n"
                "- If the function is set to maximize a likelihood or score function (e.g., log-likelihood, accuracy), respond with 'maximize'.\n"
                "- Note: Maximizing the log-probability typically corresponds to minimizing the negative log-likelihood.\n\n"
                "Examples:\n"
                "Loss Function: Cross-Entropy Loss\n"
                '{"answer": "minimize"}\n'
                "Loss Function: Custom Score Function\n"
                '{"answer": "maximize"}\n\n'
                f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )
            loss_obj_response = _query_llm("", loss_objective_prompt)
            if not loss_obj_response:
                print(f"No response for loss function objective for {loss_name}.")
                continue
            loss_obj_type = loss_obj_response.lower()

            # Instantiate the appropriate loss function instance.
            if loss_obj_type == "minimize":
                objective_function_instance = _instantiate_generic_class(ontology.MinObjectiveFunction, full_context)
            elif loss_obj_type == "maximize":
                objective_function_instance = _instantiate_generic_class(ontology.MinObjectiveFunction, full_context)
            else:
                print(f"Invalid response for loss function objective for {loss_name}.")
                continue
            new_context = full_context + [objective_function_instance]

            # Instantiate the loss function instance.
            loss_function_instance = _instantiate_cls(ontology.LossFunction, loss_name)

            # Instantiate the generic cost function instance.
            cost_function_instance = _instantiate_generic_class(ontology.CostFunction, new_context)

            # Link the loss function instance to the objective function instance.
            _link_instances(objective_function_instance, cost_function_instance, ontology.hasCost)

            # Link the loss function instance to the cost instance.
            _link_instances(cost_function_instance, loss_function_instance, ontology.hasLoss)



            # Process the regularizer functions explicitly associated with this loss function.
            regularizer_function_prompt = (
                f"Extract only the names of explicit regularizer functions that are mathematically added to the objective function for the {loss_name} loss function. "
                "Exclude implicit regularization techniques like Dropout, Batch Normalization, or any regularization that is not directly part of the loss function. "
                "Return the result in JSON format with the key 'answer'. Follow the examples below.\n\n"
                
                "Examples:\n"
                "Loss Function: Discriminator Loss\n"
                '{"answer": ["L1 Regularization"]}\n\n'
                
                "Loss Function: Generator Loss\n"
                '{"answer": ["L2 Regularization", "Elastic Net"]}\n\n'
                
                "Loss Function: Cross-Entropy Loss\n"
                '{"answer": []}\n\n'
                
                "Loss Function: Binary Cross-Entropy Loss\n"
                '{"answer": ["L2 Regularization"]}\n\n'
                
                f"Now, for the following loss function:\nLoss Function: {loss_name}\n"
                '{"answer": "<Your Answer Here>"}'
            )

            regularizer_response = _query_llm("", regularizer_function_prompt)
            if not regularizer_response:
                print(f"No response for regularizer function classes for loss function {loss_name}.")
                continue
            if regularizer_response == []:
                print(f"No regularizer functions provided for loss function {loss_name}.")
                continue
            regularizer_names = regularizer_response  # expected to be a list of regularizer names

            for reg_name in regularizer_names:
                reg_instance = _instantiate_cls(ontology.RegularizerFunction, reg_name)
                # Link the regularizer function instance to the loss function instance.
                _link_instances(cost_function_instance, reg_instance, ontology.hasRegularizer)


    
    def _process_layers(full_context: list[Thing]):

        # Get the network instance from the full context.
        network_thing = _find_ancestor_network_instance(full_context)
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        
        network_thing_name = network_thing.name

        # Process the input layer

        # Process the output layer

        # Process the hidden layers




        # Process Activation Layers
        #  Data prop has_bias (boolean)

        activation_layer_prompt = (
            "Extract the number of instances of each core layer type in the given network architecture. "
            "Only count layers that represent essential network operations such as convolutional layers, "
            "fully connected (dense) layers, and attention layers. Do NOT count layers that serve as noise layers, "
            "activation functions (e.g., ReLU, Sigmoid), or modification layers (e.g., dropout, batch normalization), "
            "or pooling layers (e.g. max pool, average pool).\n\n"
            
            "Please provide the output in JSON format using the key \"answer\", where the value is a dictionary "
            "mapping the layer type names to their counts.\n\n"
            
            "Examples:\n\n"

            "1. Network Architecture Description:\n"
            "- 3 Convolutional layers\n"
            "- 2 Fully Connected layers\n"
            "- 2 Recurrent layers\n"
            "- 1 Attention layer\n"
            "- 3 Transformer Encoder layers\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"Convolutional\": 3,\n"
            "    \"Fully Connected\": 2,\n"
            "    \"Recurrent\": 2,\n"
            "    \"Attention\": 1\n"
            "    \"Transformer Encoder\": 3\n"

            "  }\n"
            "}\n\n"

            "2. Network Architecture Description:\n"
            "- 3 Convolutional layers\n"
            "- 2 Fully Connected layer\n"
            "- 2 Recurrent layer\n"
            "- 1 Attention layers\n"
            "- 3 Transformer Encoder layers\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": {\n"
            "    \"Convolutional\": 4,\n"
            "    \"FullyConnected\": 1,\n"
            "    \"Recurrent\": 2,\n"
            "    \"Attention\": 1\n"
            "    \"Transformer Encoder\": 3\n"
            "  }\n"
            "}\n\n"

            "Now, for the following network:\n"
            f"Network: {network_thing_name}\n"
            "Expected JSON Output:\n"
            "{\n"
            "  \"answer\": \"<Your Answer Here>\"\n"
            "}\n"
        )

        activation_layer_response = _query_llm("", activation_layer_prompt)
        if not activation_layer_response:
            print("No response for activation layer classes.")
            return
        activation_layer_counts = activation_layer_response  # expected to be a dictionary of layer type names and counts

        # Process the activation layers
        for layer_type, layer_count in activation_layer_counts.items():

            # Instantiate the activation layer instances.
            for i in range(layer_count):
                layer_instance = _instantiate_cls(ontology.ActivationLayer, f"{layer_type} {i + 1}")

                print (f"Processing {layer_type} {i + 1}", type(layer_instance))

                # Convert index to ordinal for natural language
                layer_ordinal = int_to_ordinal(i + 1)

                # Link the activation layer instance to the network instance.
                _link_instances(network_thing, layer_instance, ontology.hasLayer)

                # Process the data properties of the activation layer.
                has_bias_prompt = (
                    f"Does the {layer_ordinal} {layer_type} layer include a bias term? "
                    "Please respond with either 'true', 'false', or an empty list [] if unknown, in JSON format using the key 'answer'.\n\n"
                    
                    "Clarification:\n"
                    "- A layer has a bias term if it adds a constant (bias) to the weighted sum before applying the activation function.\n"
                    "- Examples of layers that often include bias: Fully Connected (Dense) layers, Convolutional layers.\n"
                    "- Some layers like Batch Normalization typically omit bias.\n\n"

                    "Examples:\n"
                    "1. Layer: Fully-Connected\n"
                    '{"answer": "true"}\n\n'

                    "2. Layer: Convolutional\n"
                    '{"answer": "true"}\n\n'

                    "3. Layer: Attention\n"
                    '{"answer": "false"}\n\n'

                    "4. Layer: UnknownLayerType\n"
                    '{"answer": []}\n\n'

                    f"Now, for the following layer:\n"
                    f"Layer: {layer_ordinal} {layer_type}\n"
                    '{"answer": "<Your Answer Here>"}'
                )

                # Query LLM for bias term
                has_bias_response = _query_llm("", has_bias_prompt)
                if not has_bias_response:
                    print(f"No response for bias term for {layer_ordinal} {layer_type}.")
                else:
                    if has_bias_response.lower() == "true":
                        layer_instance.has_bias = [True]

                    if has_bias_response.lower() == "false":
                        layer_instance.has_bias = [False]

                    print(f"Set bias term for {layer_ordinal} {layer_type} to {layer_instance.has_bias}.")


                # Process the activation function of the activation layer

                # Convert index to ordinal for natural language
                layer_ordinal = int_to_ordinal(i + 1)

                # Process the activation function of the activation layer
                activation_function_prompt = (
                    f"Goal:\n"
                    f"Identify the activation function used in the {layer_ordinal} {layer_type} layer, if any.\n\n"

                    "Return Format:\n"
                    "Respond with the activation function name in JSON format using the key 'answer'. If there is no activation function or it's unknown, return an empty list [].\n"
                    "Example formats:\n"
                    '{"answer": "ReLU"}\n'
                    '{"answer": "Sigmoid"}\n'
                    '{"answer": []}\n\n'

                    "Clarification:\n"
                    "- An activation function is a mathematical function applied to the output of a neuron or layer in a neural network to introduce non-linearity.\n"
                    "- Common activation functions include ReLU, Sigmoid, Tanh, LeakyReLU, and Softmax.\n"
                    "- Some layers may not have an activation function explicitly applied.\n\n"

                    "Examples:\n"
                    "1. Layer: 1st Convolutional\n"
                    '{"answer": "ReLU"}\n\n'

                    "2. Layer: 2nd Fully Connected\n"
                    '{"answer": "Sigmoid"}\n\n'

                    "3. Layer: 3rd Attention\n"
                    '{"answer": []}\n\n'

                    f"Now, for the following layer:\n"
                    f"Layer: {layer_ordinal} {layer_type}\n"
                    '{"answer": "<Your Answer Here>"}'
                )

                # Query LLM for activation function
                activation_function_response = _query_llm("", activation_function_prompt)
                if not activation_function_response:
                    print(f"No response for activation function for {layer_ordinal} {layer_type}.")
                    continue
                else:
                    # Parse response
                    activation_function_name = activation_function_response

                    # Instantiate the activation function instance if found
                    if activation_function_name != "[]":
                        activation_function_instance = _instantiate_cls(ontology.ActivationFunction, activation_function_name)
                        # Link the activation function to the layer
                        _link_instances(layer_instance, activation_function_instance, ontology.hasActivationFunction)
                    else:
                        print(f"No activation function associated with {layer_ordinal} {layer_type}.")

        # Process Aggregation Layers

        # Process Modification Layers

        # Process Separation Layers
    
    def _process_task_characterization(full_context: list[Thing]):

        # Get the network instance from the full context.
        network_thing = _find_ancestor_network_instance(full_context)
        if not network_thing:
            raise ValueError("No network instance found in the full context.")
        network_thing_name = network_thing.name
        
        # Process the task characterization
        task_characterization_prompt = (
            "Extract the primary task that the given network architecture is designed to perform. "
            "The primary task is the most important or central objective of the network. "
            "Return the task name in JSON format with the key 'answer'.\n\n"

            "The types of tasks include:\n"
            "- Adversarial\n"
            "- Classification\n"
            "- SemiSupervised Classification\n"
            "- Supervised Classification\n"
            "- Clustering\n"
            "- Discrimination\n"
            "- Generation\n"
            "- Reconstruction\n"
            "- Regression\n"

            "Examples:\n"
            "1. Discrimination\n"
            '{"answer": "Discriminator"}\n\n'
            
            "2. Network: Generator\n"
            '{"answer": "Generation"}\n\n'
            
            "3. Network: Linear Regression\n"
            '{"answer": "Regression"}\n\n'
            
            "Now, for the following network:\n"
            f"Network: {network_thing_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )

        task_characterization_response = _query_llm("", task_characterization_prompt)
        if not task_characterization_response:
            print("No response for task characterization.")
            return
        task_characterization_name = task_characterization_response
        print(f"Task characterization: {task_characterization_name}", type(task_characterization_name))
        # Instantiate the task characterization instance
        task_characterization_instance = _instantiate_cls(ontology.TaskCharacterization, task_characterization_name)

        # Link the task characterization instance to the network instance
        _link_instances(network_thing, task_characterization_instance, ontology.hasTaskType)



    def _process_entity(cls: ThingClass, processed: set, full_context: list[Thing]):
        if cls in processed or cls.name in OMIT_CLASSES:
            return
        processed.add(cls)

        # if cls is ontology.Layer:
        #     _process_layers(full_context)

        # if cls is ontology.ObjectiveFunction:
        #     _process_objective_functions(full_context)

        if cls is ontology.TaskCharacterization:
            _process_task_characterization(full_context)

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
    
    
    # Verify required root exists
    if not hasattr(ontology, 'ANNConfiguration'):
        print("Error: Class 'ANNConfiguration' not found in ontology.")
        return
    
    # Initialize the LLM engine with the document context
    json_file_path = "data/alexnet/doc_alexnet.json"
    init_engine(json_file_path)

    # Extract the title from the JSON file
    from json import load
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = load(file)
    titles = [item['metadata']['title'] for item in data if 'metadata' in item and 'title' in item['metadata']]
    title = titles[0] if titles else None

    processed = set()
    processed.add(ontology.TrainingStrategy) # Temporary fix to avoid infinite loop
    
    # Create the root instance
    processed.add(ontology.ANNConfiguration)
    ann_config_instance = _instantiate_cls(ontology.ANNConfiguration, title)
    full_context = [ann_config_instance]
    llm_context = []
    
    # Process the top-level Network class.
    if hasattr(ontology, "Network"):
        network_instance = _instantiate_cls(ontology.Network, "Convolutional Network")
        object_property = ontology.hasNetwork
        # Link the network instance to the root instance.
        _link_instances(ann_config_instance, network_instance, object_property)
        new_context = full_context + [network_instance]
        new_llm_context = llm_context + [network_instance.name]
        for connected_class in get_connected_classes(ontology.Network, ontology):
            if isinstance(connected_class, ThingClass):
                _process_entity(connected_class, processed, new_context, new_llm_context)
    
    print("An ANN has been successfully instantiated.")

if __name__ == "__main__":
    OUTPUT_FILE = './test.txt'
    ontology_path = f"./data/owl/{C.ONTOLOGY.FILENAME}"
    ontology = get_ontology(ontology_path).load()
    
    with ontology:
        dfs_instantiate_annetto(ontology)
        new_file_path = "data/owl/annett-o-test.owl"
        ontology.save(file=new_file_path, format="rdfxml")
        print(f"Ontology saved to {new_file_path}")
