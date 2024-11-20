# from langchain.graphs import OntotextGraphDB
from langchain.chains import OntotextGraphDBQAChain


class OntologyHandler:
    def __init__(self, endpoint_url, repository_name, llm_model="llama3"):
        print("Connecting to GraphDB...")
        
        # Initialize the LLM for SPARQL generation and QA
        llm = OllamaLLM(model=llm_model)

        # Define the prompt templates for SPARQL generation and fixes
        sparql_prompt_template = PromptTemplate(
            input_variables=["query"],
            template="Generate a SPARQL query for: {query}",
        )
        sparql_fix_template = PromptTemplate(
            input_variables=["query", "error"],
            template="Fix the SPARQL query: {query}.\nError: {error}",
        )

        # Create SPARQL generation and fix chains
        sparql_generation_chain = LLMChain(llm=llm, prompt=sparql_prompt_template)
        sparql_fix_chain = LLMChain(llm=llm, prompt=sparql_fix_template)

        # Create OntotextGraphDB client
        self.graphdb = OntotextGraphDB(
            endpoint_url=endpoint_url,
            repository_name=repository_name,
            sparql_generation_chain=sparql_generation_chain,
            sparql_fix_chain=sparql_fix_chain,
        )
        print("Connected to GraphDB.")

    def load_ontology(self, ontology_file):
        print("Loading ontology into GraphDB...")
        with open(ontology_file, "r") as file:
            ontology_data = file.read()
        self.graphdb.store_ontology(ontology_data)
        print("Ontology loaded into GraphDB.")

    def query_ontology(self, query):
        print("Querying GraphDB...")
        result = self.graphdb.query(query)
        print("Query result obtained.")
        return result

# class OntologyHandler:
#     def __init__(self, endpoint_url, repository_name):
#         print("Connecting to GraphDB...")
#         self.graphdb = OntotextGraphDBQAChain(endpoint_url=endpoint_url, repository_name=repository_name)
#         print("Connected to GraphDB.")

#     def load_ontology(self, ontology_file):
#         print("Loading ontology into GraphDB...")
#         with open(ontology_file, "r") as file:
#             ontology_data = file.read()
#         self.graphdb.store_ontology(ontology_data)
#         print("Ontology loaded into GraphDB.")

#     def query_ontology(self, query):
#         print("Querying GraphDB...")
#         result = self.graphdb.query(query)
#         print("Query result obtained.")
#         return result
