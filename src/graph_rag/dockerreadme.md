```markdown
# 🧠 Neo4j + n10s + NLM Ingestor Docker Environment

This project sets up a lightweight semantic stack using Docker Compose with:

- 🕸️ **Neo4j Community Edition 5.20** with:
  - RDF/OWL support via the [n10s (Neosemantics)](https://neo4j.com/labs/neosemantics/) plugin
  - Import utilities via [APOC Extended](https://neo4j.com/labs/apoc/)
- 🧾 **NLM Ingestor** for biomedical document processing
- 🐍 **Python utility** (`sample.py`) for automated OWL ontology import

---

## 📦 Services

### 🧩 Neo4j

Configured to support:

- RDF/OWL file ingestion via the `n10s` plugin
- APOC extended plugin
- Custom OWL imports from the mounted `/import` directory

**Ports:**
- `7474`: Neo4j browser
- `7687`: Bolt protocol (used by Python/clients)

**Volumes:**
- `./neo4j_data`: Persistent graph data
- `./import`: OWL/RDF files (shared with Neo4j and Python)

### 📄 NLM Ingestor

Biomedical ingestion API served on:

- `5001`: NLM Ingestor REST endpoint

---

## 🐍 `sample.py`: Automated OWL Import Script

This Python script connects to the Neo4j instance and automates a complete RDF/OWL import cycle.

### 🔧 Features:
- Connects using `neo4j` Python driver
- Deletes all existing nodes
- Drops & re-creates uniqueness constraints
- Initializes the `n10s` graph configuration
- Loads OWL data from the `/import` directory

### 📂 File structure:

```
.
├── docker-compose.yml
├── sample.py                # This script
├── import/
│   └── annett-o-0.1.owl     # OWL file to be imported
└── neo4j_data/              # Persistent Neo4j volume
```

### ▶️ Usage:

Make sure the services are running:

```bash
docker-compose up -d
```

Then run the script:

```bash
python sample.py
```

### 📥 What it does:

1. Clears the graph:
   ```cypher
   MATCH(n) DETACH DELETE n;
   ```
2. Drops and re-creates the `n10s_unique_uri` constraint.
3. Initializes graph config:
   ```cypher
   CALL n10s.graphconfig.init({
     handleVocabUris: "SHORTEN",
     keepLangTag: false,
     handleMultival: "ARRAY"
   });
   ```
4. Loads the OWL file:
   ```cypher
   CALL n10s.rdf.import.fetch("file:///import/annett-o-0.1.owl", "RDF/XML");
   ```

### 📤 Sample Output:

```bash
Import summary: <neo4j.ResultSummary.counters>
```

You can now query your ontology using Cypher via the Neo4j browser at [http://localhost:7474](http://localhost:7474).

---

## 📋 Example Query

```cypher
MATCH (c:Class) RETURN c LIMIT 10;
```

or find all relationships from ANNConfiguration:

```cypher
MATCH (n:ns0__ANNConfiguration)-[r]->(m) RETURN n, type(r), m;
```

---

## 📬 Questions?

For more on the tools:

- [n10s GitHub](https://github.com/neo4j-labs/neosemantics)
- [APOC GitHub](https://github.com/neo4j-contrib/neo4j-apoc-procedures)
- [NLM Ingestor GitHub](https://github.com/NLMatics/nlm-ingestor)

---
```

Sure! Here's the **📦 pip install** section you can append to the README:

---

## 📦 Python Setup

To run `sample.py`, you need Python 3.7+ and the Neo4j Python driver installed.

### 🔧 Install dependencies:

```bash
pip install neo4j
```

If you're using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install neo4j
```


