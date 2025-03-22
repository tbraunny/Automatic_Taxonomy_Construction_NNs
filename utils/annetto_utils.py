from re import sub, fullmatch
from owlready2 import ThingClass
from typing import List, Union, Optional
from rapidfuzz import process, fuzz


def int_to_ordinal(n):

    """
    Convert an integer into its ordinal representation.
    For example: 1 -> "1st", 2 -> "2nd", 3 -> "3rd", 4 -> "4th", etc.
    """
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def split_camel_case(names:Union[list[str], str]) -> list[str]:

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
    
def _unhash_and_format_instance_name(self, instance_name: str) -> str:
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
        return stripped_name.replace("-", " ")  # Convert dashes back to spaces