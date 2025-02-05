import json

class ConversationTree:
    """
    Manages and navigates a hierarchical conversation tree.
    """

    def __init__(self):
        # Updated: Replace 'qa_dict' with 'question' and 'answer'
        self.tree = {
            "id": 0,
            "cls_name": None,
            "question": None,
            "answer": None,
            "children": [],
            "parent_id": None
        }
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}

    def add_child(self, parent_id, cls_name, question=None, answer=None):
        """
        Adds a child node to the conversation tree.

        :param parent_id: ID of the parent node.
        :param cls_name: The class name.
        :param question: The question asked (optional).
        :param answer: The answer received (optional).
        :return: The new node's ID.
        """

        self.node_id += 1

        child_node = {
            "id": self.node_id,
            "cls_name": cls_name,
            "question": question, 
            "answer": answer,     
            "children": [],
            "parent_id": parent_id  
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
        from datetime import datetime
        import json
        """
        Saves the conversation tree to a JSON file.

        :param file_name: Base name for the file where the tree will be saved.
        """
        # Get the current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Append the timestamp to the file name
        file_name_with_timestamp = f"{file_name}_{timestamp}.json"

        # Save the JSON file
        with open(file_name_with_timestamp, "w") as f:
            json.dump(self.to_serializable(self.tree), f, indent=4)
        print(f"Saved file as {file_name_with_timestamp}")

    
    """ Helpers """

    def get_parent_cls_name(self, node_id):
        """
        Retrieves the 'cls_name' value of the parent node for a given node.

        Args:
            node_id (int): The ID of the node whose parent's 'cls_name' is to be retrieved.

        Returns:
            The 'cls_name' of the parent node if it exists, or None if the node has no parent or the parent doesn't have a 'cls_name'.
        """
        # Get the current node
        current_node = self.nodes.get(node_id)
        if not current_node:
            print(f"Node with ID {node_id} does not exist.")
            return None

        # Get the parent ID
        parent_id = current_node.get("parent_id")
        if parent_id is None:
            print(f"Node with ID {node_id} has no parent.")
            return None

        # Get the parent node
        parent_node = self.nodes.get(parent_id)
        if not parent_node:
            print(f"Parent node with ID {parent_id} does not exist.")
            return None

        # Return the parent's 'cls_name'
        return parent_node.get("cls_name")


