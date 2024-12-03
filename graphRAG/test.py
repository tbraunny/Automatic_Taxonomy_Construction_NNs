from neo4j import GraphDatabase

uri = "bolt://172.24.208.1:7687" 
username = "neo4j"
password = "Sakul32524!"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Example query to test the connection
with driver.session() as session:
    result = session.run("RETURN 'Hello, Neo4j!' AS message")
    for record in result:
        print(record["message"])
