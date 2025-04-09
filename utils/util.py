
def contains_negative_words(text: str) -> bool:
    keywords = {"no", "none", "null"}
    return any(word in text.lower().split() for word in keywords)

def get_sanitized_attr(obj, attr_path: str) -> str | None:
    """
    Safely gets an attribute via a dot-separated path like 'answer.task_type.name',
    and returns None if the value is a string and contains a negative word.
    """
    try:
        value = obj
        for attr in attr_path.split("."):
            value = getattr(value, attr)
        if isinstance(value, str) and contains_negative_words(value):
            return None
        return value
    except AttributeError:
        print(f"Attribute Error '{attr_path}' not found in the object.")
        return None
