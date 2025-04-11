import tests.papers_rag.instantiate_annetto as inst_ann
import  tests.deprecated.llm_service as llm_service

from owlready2 import *
from unittest.mock import MagicMock, patch

import src.instantiate_annetto.instantiate_annetto as inst_ann
from utils.constants import Constants as C
import traceback
from src.instantiate_annetto.instantiate_annetto import OntologyInstantiator


def test_num_tokens_from_string():
    assert inst_ann.num_tokens_from_string("a b c") == 3

def test_query_llm():
    model_name = "alexnet"
    network_instance_name = "convolutional network"
    try:
        llm_service.init_engine(model_name, f"data/{model_name}/doc_{model_name}.json")
    except Exception as e:
        raise Exception(f"Error initializing the LLM engine: {e}")

    try:
        int_prompt = (
            f"What is the number of input parameters in the {network_instance_name} network architecture?\n"
            "Return the result as an integer in JSON format with the key 'answer'.\n\n"
            "Examples:\n"
            "1. Network: Discriminator\n"
            '{"answer": 1}\n\n'
            "2. Network: Generator\n"
            '{"answer": 784}\n\n'
            "3. Network: Linear Regression\n"
            '{"answer": 1}\n\n'
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
            )
        assert type(llm_service.query_llm(model_name,int_prompt)) == int
    except Exception as e:
        raise Exception(f"LLM Service failed to response with a propert integer json format: {e}")

    try:
        str_prompt = (
            f"Goal:\Name the first layer used in the {network_instance_name} network.\n\n"
            "Return Format:\nRespond with the layer name in JSON format using the key 'answer'. If there is no activation function or it's unknown, return an empty list [].\n"
            "Examples:\n"
            '{"answer": "Convolutional"}\n'
            '{"answer": "Dense"}\n'
            '{"answer": []}\n\n'
            f"Now, for the following network:\Network: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        assert type(llm_service.query_llm(model_name,str_prompt)) == str
    except Exception as e:
        raise Exception(f"LLM Service failed to response with a propert string json format: {e}")

    # Process Activation Layers
    try:
        dict_prompt = (
            f"Extract the number of instances of convolutional layers and fully-connected (dense) layers in the {network_instance_name} architecture. "
            'Please provide the output in JSON format using the key "answer", where the value is a dictionary '
            "mapping the layer type name to it's counts.\n\n"
            "Examples:\n\n"
            "Expected JSON Output:\n"
            "{\n"
            '  "answer": {\n'
            '    "Convolutional": 4,\n'
            '    "FullyConnected": 1,\n'
            "  }\n"
            "}\n\n"
            "Now, for the following network:\n"
            f"Network: {network_instance_name}\n"
            "Expected JSON Output:\n"
            "{\n"
            '  "answer": "<Your Answer Here>"\n'
            "}\n"
        )
        respose =  llm_service.query_llm(model_name,dict_prompt)
        assert isinstance(respose, dict) and all(isinstance(k, str) and isinstance(v, any) for k, v in respose.items())
    except Exception as e:
        raise Exception(f"LLM Service failed to response with a propert dictionary json format: {e}")
    
    try:
        list_of_any_prompt = (
            f"List the first two layers used in the {network_instance_name} network architecture.\n\n"
            "Return Format:\nRespond with a list of activation function names in JSON format using the key 'answer'.\n"
            "Examples:\n"
            '{"answer": ["ReLU", "Sigmoid"]}\n'
            '{"answer": ["ReLU", "Softmax"]}\n'
            '{"answer": ["ReLU"]}\n\n'
            f"Now, for the following network:\nNetwork: {network_instance_name}\n"
            '{"answer": "<Your Answer Here>"}'
        )
        response = llm_service.query_llm(model_name,list_of_any_prompt)

            if result != mock_instance:
                errors.append(
                    "Returned instance does not match expected mock instance."
                )

            if not instance._hash_and_format_instance_name.called:
                errors.append("_hash_and_format_instance_name() was not called.")

            if not mock_create_instance.called:
                errors.append("create_cls_instance() was not called.")

            if not any("Instantiated MockThingClass" in log for log in log_calls):
                errors.append("Logging message for instance creation not found.")

            print("\nTest Debugging Output:")
            print(f"Mock Instance Created: {result}")
            print(f"Log Calls: {log_calls}")

    except Exception as e:
        errors.append(f"Exception occurred: {e}")
        print("\nTest Failed with Exception:")
        traceback.print_exc()

    if errors:
        print("\nFAILED ")
        print(f"Total Errors: {len(errors)}")
        for error in errors:
            print(error)
    else:
        print("\nPASSED")


test_instantiate_and_format_class()
