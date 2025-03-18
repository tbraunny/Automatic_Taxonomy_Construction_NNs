from re import sub, fullmatch
from owlready2 import ThingClass, Ontology, get_ontology
from typing import List, Union, Optional
from rapidfuzz import process, fuzz

from utils.constants import Constants as C

def load_annetto_ontology(release_type:str)->Ontology:
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
        "stable": C.ONTOLOGY.STABLE_ONTOLOGY_PATH
    }

    if release_type not in releases:
        raise ValueError(f"Invalid release type: {release_type}. Must be one of {list(releases)}")

    return get_ontology(releases[release_type]).load()

def int_to_ordinal(n:int)->str:
    """
    Convert an integer into its ordinal representation.
    For example: 1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th", etc.

    :param n: The integer to convert
    :return: The ordinal representation
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def split_camel_case(names:Union[list[str], str]) -> list[str]:
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
        if fullmatch(r'[A-Z]{2,}[a-z]*$', name):  # Skip all-uppercase acronyms like "RNRtop"
            split_names.append(name)
        else:
            # Split between lowercase-uppercase (e.g., "NoCoffee" → "No Coffee")
            name = sub(r'([a-z])([A-Z])', r'\1 \2', name)

            # Split when a sequence of uppercase letters is followed by a lowercase letter
            # (e.g., "CNNModel" → "CNN Model")
            name = sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)

            split_names.append(name)

    return split_names

def thingclass_names_to_str(list_things:List[ThingClass]) -> List[str]:
    """
    Takes a list of ThingClass objects and returns a list of their names as strings.
    
    :param things: List of ThingClass objects
    :return: List of names as strings
    """
    return [thing.name for thing in list_things]

def comma_separate(strings:List[str]) -> str:
    """
    Takes a list of strings and returns a single comma-separated string.
    
    :param strings: List of strings
    :return: Comma-separated string
    """
    return ", ".join(strings)

def make_thing_classes_readable(things:List[ThingClass]) -> str:
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

if __name__ == "__main__":
    ontology = load_annetto_ontology("meow")