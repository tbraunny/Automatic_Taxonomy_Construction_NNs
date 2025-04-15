
# Frontend 3/18/2025

There are three different applications used for the frontend: streamlit, neo4j, and voila.

## Description

**streamlit**: Offers a way to turn python code into a front end application.

**neo4j**: Acts as a graph database for graphRAG, llm inference, and data.

**voila**: Server for the graph visuals connected to neo4j.

## Getting Started

### Dependencies

- **Core Dependencies**:
  - `streamlit`
  - `neo4j`
  - `voila`

### Installing
This guide is for installation on linux

- **streamlit**: [Installation Guide](https://docs.streamlit.io/get-started/installation)
```bash
pip install streamlit
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

MERGE (taskType_Taxon:Taxonomy {name: 'TaskType', type: 'CategoryTask'})
MERGE (FullyConnectedLayerType_Taxon:Taxonomy {name: 'LayerType', type: 'CategoryLayer', LayerTypes: 'FullyConnected'})
MERGE (CompositeLayerType_B_C_D_FC_Taxon:Taxonomy {name: 'LayerType', type: 'CategoryLayer', LayerTypes: 'BatchNorm, Concat, Dropout, FullyConnected'})
MERGE (GAN_Taxon:Taxonomy {name: 'GAN', type: 'Network', task: 'Adversarial, Classification, Generative', LayerTypes: 'FullyConnected'})
MERGE (AAE_Taxon:Taxonomy {name: 'AAE', type: 'Network', task: 'Adversarial, Clustering, Discrimination, Generation, Reconstruction', LayerTypes: 'BatchNorm, Concat, Dropout, FullyConnected'})
MERGE (simpleClassification_Taxon:Taxonomy {name: 'SimpleClassification', type: 'Network', task: 'SupervisedClassification', LayerTypes: 'FullyConnected'})
MERGE (taskType_Taxon)-[:hasAdversarial]->(FullyConnectedLayerType_Taxon)
MERGE (taskType_Taxon)-[:hasClassification]->(FullyConnectedLayerType_Taxon)
MERGE (taskType_Taxon)-[:hasGenerative]->(FullyConnectedLayerType_Taxon)
MERGE (taskType_Taxon)-[:hasSupervisedClassification]->(FullyConnectedLayerType_Taxon)
MERGE (taskType_Taxon)-[:hasAdversarial]->(CompositeLayerType_B_C_D_FC_Taxon)
MERGE (taskType_Taxon)-[:hasClustering]->(CompositeLayerType_B_C_D_FC_Taxon)
MERGE (taskType_Taxon)-[:hasDiscrimination]->(CompositeLayerType_B_C_D_FC_Taxon)
MERGE (taskType_Taxon)-[:hasGeneration]->(CompositeLayerType_B_C_D_FC_Taxon)
MERGE (taskType_Taxon)-[:hasReconstruction]->(CompositeLayerType_B_C_D_FC_Taxon)
MERGE (CompositeLayerType_B_C_D_FC_Taxon)-[:hasBatchNorm]->(AAE_Taxon)
MERGE (CompositeLayerType_B_C_D_FC_Taxon)-[:hasConcat]->(AAE_Taxon)
MERGE (CompositeLayerType_B_C_D_FC_Taxon)-[:hasDropout]->(AAE_Taxon)
MERGE (CompositeLayerType_B_C_D_FC_Taxon)-[:hasFullyConnected]->(AAE_Taxon)
MERGE (FullyConnectedLayerType_Taxon)-[:hasFullyConnected]->(GAN_Taxon)
MERGE (FullyConnectedLayerType_Taxon)-[:hasFullyConnected]->(simpleClassification_Taxon)

```
*Note: ensure paths all paths are correct according to your local machine. The cypher merge is to visualize a tertiary taxonomy*

- ollama: [PyPI Page](https://pypi.org/project/ollama/)  
  *Currently uses the model `llama3.1:8b-instruct-fp16`, but can be changed in the source code.*

- **Voila**:
  ```bash
  pip install voila
  ```
  Requires `yfiles_jupyter_graphs` for visualization  
  [yFiles Jupyter Graphs GitHub](https://github.com/yWorks/yfiles-jupyter-graphs)
  ```bash
  pip install yfiles_jupyter_graphs
  ```


### Program Details
start the program in this order

- **neo4j**
```bash
sudo systemctl start neo4j
```

- **voila**
```bash
cd Automatic_Taxonomy_Construction_NNs/front_end/voila
```
```bash
voila yfiles_graph.ipynb --Voila.tornado_settings="{'headers':{'Content-Security-Policy':'frame-ancestors *'}}"
```
- **streamlit**
```bash
cd Automatic_Taxonomy_Construction_NNs/front_end/streamlit
```
```bash
streamlit run app.py
```





