# GraphRAG 3/18/2025

GraphRAG (Graph-based Retrieval-Augmented Generation) enhances language model performance by combining graph-based knowledge representation with retrieval-augmented generation. It is ideal for tasks requiring external knowledge integration, such as answering complex questions, summarizing information, or generating factually accurate responses.

## Description

GraphRAG serves as a knowledge base for a language model (LLM) to infer relationships within an ontology or to derive a schema for a taxonomy.

## Getting Started

The README is structured as follows:
1. Python script (`.py`) instructions first
2. Jupyter Notebook (`.ipynb`) instructions follow

### Dependencies

- **Core Dependencies**:
  - `langchain`
  - `neo4j`
  - `ollama`

- **Additional Dependencies** (for running `FinalGraphRAG.ipynb`):
  - Jupyter Notebook
  - `yfiles_jupyter_graphs` for graph visualization

### Installing
This guide is for installation on linux

- **langchain**: [Installation Guide](https://python.langchain.com/docs/introduction/)
```bash
pip install langchain-core langchain-neo4j langchain-community langchain-experimental langchain-ollama neo4j
```

- **neo4j**: [Installation Guide](https://neo4j.com/docs/operations-manual/current/installation/) 

```bash
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
```

```bash, specify your version
sudo apt-get install neo4j=1:2025.02.0
```

*Note: The APOC and Neosemantics plugins are required.*
  
Since APOC and Neosemantics relies on Neo4j’s internal APIs you need to use the matching APOC version for your Neo4j installation. Make sure that the first two version numbers match between Neo4j and APOC.

[Apoc plugin download](https://github.com/neo4j/apoc/releases/)

[Neosemantics plugin download](https://github.com/neo4j-labs/neosemantics/releases)

these will both be .jar files and will need to be placed in this directiory: /var/lib/neo4j/plugins/

You will then need to add configuration for them by adding the following lines to /etc/neo4j/neo4j.conf at the bottom of the file

```/etc/neo4j/neo4j.conf
dbms.security.auth_minimum_password_length=1 (allows the default password to be neo4j)
server.unmanaged_extension_classes=n10s.endpoint=/rdf
dbms.security.procedures.unrestricted=apoc.*, n10s.*
dbms.security.procedures.whitelist=apoc.*, n10s.*, apoc.load.*,apoc.coll.*
dbms.directories.import=Automatic_Taxonomy_Construction_NNs/data
```
*Note: ensure paths all paths are correct according to your local machine.*

to start the server, restart, and get the status of the server or start a cyphershell:

```bash
sudo systemctl start neo4j
sudo systemctl status neo4j
sudo systemctl restart neo4j
sudo cypher-shell
```

to import rdf/xml data use the following Cypher queries in cypher-shell:
```Cypher
MATCH(n) DETACH DELETE n;

DROP CONSTRAINT n10s_unique_uri;

CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

CALL n10s.graphconfig.init({
    handleVocabUris: "SHORTEN",
    keepLangTag: false,
    handleMultival: "ARRAY"
});

CALL n10s.rdf.import.fetch("file:///Automatic_Taxonomy_Construction_NNs/data/Annett-o/annett-o-0.1.owl", "RDF/XML");
```
*Note: ensure paths all paths are correct according to your local machine.*

- **ollama**: [PyPI Page](https://pypi.org/project/ollama/)  
  *Currently uses the model `llama3.1:8b-instruct-fp16`, but can be changed in the source code.*

- **Jupyter Notebook**:  
  Requires `yfiles_jupyter_graphs` for visualization  
  [yFiles Jupyter Graphs GitHub](https://github.com/yWorks/yfiles-jupyter-graphs)
   ```bash
  pip install yfiles_jupyter_graphs
  ```


### Program Details

- **graphRAG.py**  
A script that prints answers to user questions based on predefined template prompts. Modifications to the question or model must be made in the source code.
cd into Automatic_Taxonomy_Construction_NNs/src/graph_rag then run:
```bash
python graphRAG.py
```
This will print out an answer from the LLM based on the cypher query template inputted, graph Schema, and question provided.

- **FinalGraphRAG.ipynb**  
  A research environment designed for testing and adding additional code to `graphRAG.py`. It offers an easier way to visualize the knowledge base.



