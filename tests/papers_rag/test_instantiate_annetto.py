from owlready2 import *
from unittest.mock import MagicMock, patch

import src.instantiate_annetto.instantiate_annetto as inst_ann
from utils.constants import Constants as C
import traceback
from src.instantiate_annetto.instantiate_annetto import OntologyInstantiator


def test_num_tokens_from_string():
    assert inst_ann.num_tokens_from_string("a b c") == 3


def test_instantiate_and_format_class():
    errors = []

    try:
        mock_cls = MagicMock()
        mock_cls.name = "MockThingClass"
        mock_instance_name = "TestInstance"

        instance = OntologyInstantiator(
            C.ONTOLOGY.FILENAME,
            [
                "/home/richw/.richie/Automatic_Taxonomy_Construction_NNs \
                                                              /data/alexnet/doc_alexnet.json"
            ],
        )

        instance._hash_and_format_instance_name = MagicMock(
            return_value="hashed_TestInstance"
        )
        instance._unhash_and_format_instance_name = MagicMock(
            return_value="TestInstance"
        )
        instance.logger = MagicMock()

        with patch("instantiate_annetto.create_cls_instance") as mock_create_instance:
            mock_instance = MagicMock()
            mock_create_instance.return_value = mock_instance

            result = instance._instantiate_and_format_class(
                mock_cls, mock_instance_name
            )

            log_calls = [call.args[0] for call in instance.logger.info.call_args_list]

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
