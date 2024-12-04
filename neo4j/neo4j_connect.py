from neo4j import GraphDatabase

URI = "neo4j://localhost:7687" # port 7687
AUTH = ("ATC_NN_neo4j", "taxonomies")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

def upload_owl():
    pass