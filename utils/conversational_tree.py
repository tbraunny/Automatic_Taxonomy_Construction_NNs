import json

class ConversationTree:
    """
    Manages and navigates a hierarchical conversation tree.
    """

    def __init__(self):
        self.tree = {"id": 0, "question": None, "answer": None, "children": [], "parent_id": None}
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}

    def add_child(self, parent_id, question, answer=None):
        """
        Adds a child node to the conversation tree.

        :param parent_id: ID of the parent node.
        :param question: The question asked.
        :param answer: The answer received (optional).
        :return: The new node's ID.
        """
        self.node_id += 1

        child_node = {
            "id": self.node_id,
            "question": question,
            "answer": answer,
            "children": [],
            "parent_id": parent_id  # Link this node to its parent
        }
        self.nodes[parent_id]["children"].append(child_node)  # Add child to parent's list
        self.nodes[self.node_id] = child_node  # Store the new node in the tree
        return self.node_id


    def to_serializable(self, obj):
        """
        Converts non-serializable objects to a JSON-compatible structure.

        :param obj: The object to convert.
        :return: A JSON-serializable structure.
        """
        from requests.models import Response
        if isinstance(obj, dict):
            return {key: self.to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.to_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self.to_serializable(vars(obj))
        elif isinstance(obj, Response):
            return {"response_text": str(obj)}
        else:
            return obj
        


    def save_to_json(self, file_name):
        """
        Saves the conversation tree to a JSON file.

        :param file_name: Path to the file where the tree will be saved.
        """
        with open(file_name, "w") as f:
            json.dump(self.to_serializable(self.tree), f, indent=4)
