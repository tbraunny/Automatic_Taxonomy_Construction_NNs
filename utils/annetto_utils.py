from re import sub, fullmatch
from owlready2 import ThingClass, Ontology, get_ontology
from typing import List, Union, Optional
from rapidfuzz import process, fuzz
import os
from utils.constants import Constants as C


def load_annetto_ontology(
    return_onto_from_path: str = None, return_onto_from_release: str = None
) -> Ontology:
    """
    Loads the Annetto ontology from the file system.
    Must choose a release type: "development", "test", "base", or "stable".

    :param: release_type: The type of release to load.
    :return: The Annett-o ontology object.
    """

    releases = {
        "development": C.ONTOLOGY.DEV_ONTOLOGY_PATH,
        "test": C.ONTOLOGY.TEST_ONTOLOGY_PATH,
        "base": C.ONTOLOGY.BASE_ONTOLOGY_PATH,
        "stable": C.ONTOLOGY.STABLE_ONTOLOGY_PATH,
        "MID_DEMO": C.ONTOLOGY.MID_DEMO_ONTOLOGY_PATH,
    }
    if return_onto_from_path and return_onto_from_release:
        raise ValueError(
            "Must provide either an owl file path or release name. Releases can be of types development, test, base, stable, or MID_DEMO."
        )

    # Check if the path exists
    if return_onto_from_path and not os.path.exists(return_onto_from_path):
        raise FileNotFoundError(
            f"Ontology path {return_onto_from_path} does not exist."
        )
    # If a path is provided, load the ontology from that path
    if return_onto_from_path:
        return get_ontology(return_onto_from_path)

    # Check if the release type is valid
    if return_onto_from_release not in list(releases.keys()):
        raise ValueError(
            f"Invalid release type: {return_onto_from_release}. Must be one of {list(releases)}"
        )
    # Load the ontology from the specified release
    return get_ontology(releases[return_onto_from_release]).load()


def int_to_ordinal(n: int) -> str:
    """
    Convert an integer into its ordinal representation.
    For example: 1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th", etc.

    :param n: The integer to convert
    :return: The ordinal representation
    """
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def split_camel_case(names: Union[list[str], str]) -> list[str]:
    """
    Split camelCase strings or list of strings into separate words.
    Accounts for acronyms (e.g., "CNNModel" → "CNN Model").

    :param names: A string or list of strings
    :return: A list of strings with the camelCase strings split into separate words
    """

    if isinstance(names, str):  # If a single string is passed, convert it into a list
        names = [names]

    split_names = []
    for name in names:
        if fullmatch(
            r"[A-Z]{2,}[a-z]*$", name
        ):  # Skip all-uppercase acronyms like "RNRtop"
            split_names.append(name)
        else:
            # Split between lowercase-uppercase (e.g., "NoCoffee" → "No Coffee")
            name = sub(r"([a-z])([A-Z])", r"\1 \2", name)

            # Split when a sequence of uppercase letters is followed by a lowercase letter
            # (e.g., "CNNModel" → "CNN Model")
            name = sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)

            split_names.append(name)

    return split_names


def thingclass_names_to_str(list_things: List[ThingClass]) -> List[str]:
    """
    Takes a list of ThingClass objects and returns a list of their names as strings.

    :param things: List of ThingClass objects
    :return: List of names as strings
    """
    return [thing.name for thing in list_things]


def comma_separate(strings: List[str]) -> str:
    """
    Takes a list of strings and returns a single comma-separated string.

    :param strings: List of strings
    :return: Comma-separated string
    """
    return ", ".join(strings)


def make_thing_classes_readable(things: List[ThingClass]) -> str:
    """
    Takes a list of ThingClass objects and returns a human-readable string with their names.

    :param things: List of ThingClass objects
    :return: Human-readable string
    """
    return comma_separate(split_camel_case(thingclass_names_to_str(things)))


def fuzzy_match_class(
    instance_name: str, classes: List[ThingClass], threshold: int = 80
) -> Optional[ThingClass]:
    """
    Perform fuzzy matching to find the best match for an instance to a known class.

    :param instance_name: The instance name.
    :param classes: A list of ThingClass objects to match with.
    :param threshold: The minimum score required for a match.
    :return: The best-matching ThingClass object or None if no good match is found.
    """
    if not instance_name or not classes:
        return None

    # Convert classes to a dictionary for lookup
    class_name_map = {cls.name: cls for cls in classes}

    match, score, _ = process.extractOne(
        instance_name, class_name_map.keys(), scorer=fuzz.ratio
    )

    return class_name_map[match] if score >= threshold else None


def unformat_instance_name(instance_name: str) -> str:
    """
    Remove the ANN config hash prefix from the instance name and restore readability.

    This method extracts the actual instance name by stripping out the
    prefixed hash and replacing dashes with spaces.

    Example: Input: "abcd1234_convolutional-layer" Output: "Convolutional Layer"

    Args:
        instance_name (str): The unique instance name with the hash prefix.

    Returns:
        str: The original readable instance name.
    """
    parts = instance_name.split("_", 1)  # Split at the first underscore
    stripped_name = parts[-1]  # Extract the actual instance name (without hash)
    stripped_name = stripped_name.replace("-", " ")  # Convert dashes back to spaces
    # capitalize the first letter of each word
    stripped_name = " ".join(word.capitalize() for word in stripped_name.split())
    return stripped_name

if __name__ == "__main__":
    ontology = load_annetto_ontology("meow")
