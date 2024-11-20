import json
# from collections import defaultdict


class ConversationTree:
    def __init__(self):
        # Initialize the tree structure
        self.tree = {"id": 0, "question": None, "answer": None, "children": []}
        self.node_id = 0
        self.current_node = self.tree
        self.nodes = {0: self.tree}  # Map of node_id to tree nodes

    def add_child(self, parent_id, question, answer=None):
        """Add a child node to the tree."""
        self.node_id += 1
        child_node = {"id": self.node_id, "question": question, "answer": answer, "children": []}
        self.nodes[parent_id]["children"].append(child_node)
        self.nodes[self.node_id] = child_node

    def update_node_answer(self, node_id, answer):
        """Update the answer for a specific node."""
        if node_id in self.nodes:
            self.nodes[node_id]["answer"] = answer

    def get_ancestor_context(self, node_id):
        """Get ancestor questions and answers for the context."""
        context = []
        current_node = self.nodes[node_id]
        while current_node.get("id") != 0:  # Stop at the root node
            context.insert(0, {
                "question": current_node["question"],
                "answer": current_node["answer"]
            })
            parent_node = next((node for node in self.nodes.values()
                                if current_node in node["children"]), None)
            if parent_node:
                current_node = parent_node
            else:
                break
        return context

    def save_to_json(self, file_name):
        """Save the tree structure to a JSON file."""
        with open(file_name, "w") as f:
            json.dump(self.tree, f, indent=4)

    def load_from_json(self, file_name):
        """Load the tree structure from a JSON file."""
        with open(file_name, "r") as f:
            self.tree = json.load(f)
            self._rebuild_node_map(self.tree)

    def _rebuild_node_map(self, node):
        """Rebuild the node map from a loaded JSON tree."""
        self.nodes = {}
        self.node_id = 0
        self._rebuild_helper(node)

    def _rebuild_helper(self, node):
        self.nodes[node["id"]] = node
        self.node_id = max(self.node_id, node["id"])
        for child in node["children"]:
            self._rebuild_helper(child)


# Example usage
if __name__ == "__main__":
    # import openai  # Replace with the library or API client you're using for LLM interaction

    # Initialize conversation tree
    tree = ConversationTree()

    # Define LLM call
    def ask_llm(question, context=None):
        """Call the LLM with a question and optional context."""
        prompt = "\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in context) if context else ""
        prompt += f"\nQ: {question}\nA:"
        # Mock response (replace with LLM response)
        response = "Mock answer for demonstration."
        return response

    # Root question
    root_question = "What networks are in a GAN?"
    root_answer = ask_llm(root_question)
    tree.add_child(0, root_question, root_answer)

    # Child questions
    parent_id = 1  # ID of the root node
    child_questions = [
        "How many layers does the generator network have?",
        "How many layers does the adversarial network have?"
    ]
    for question in child_questions:
        context = tree.get_ancestor_context(parent_id)
        answer = ask_llm(question, context)
        tree.add_child(parent_id, question, answer)

    # Save conversation to JSON
    tree.save_to_json("conversation_tree.json")
