from neo4j import GraphDatabase

# Connect to your Neo4j instance
uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j"

driver = GraphDatabase.driver(uri, auth=(username, password))

def import_rdf(session, url, rdf_format="RDF/XML", options=None):
    if options is None:
        options = {}

    cypher_query = """
    CALL n10s.rdf.import.fetch($url, $format, $options)
    """
    setup_query = """
    CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
    CALL n10s.graphconfig.init()
    """
    #result = session.run(setup_query, url=url)
    result = session.run(cypher_query, url=url, format=rdf_format, options=options)
    summary = result.consume()
    return summary.counters

if __name__ == '__main__':
    with driver.session() as session:
        file_path = "file:///import/annetto-sample.owl"
        rdf_format = "RDF/XML"
        options = {
            "handleVocabUris": "SHORTEN",
            "typesToLabels": True,
            "commitSize": 1000
        }

        counters = import_rdf(session, file_path, rdf_format, options)
        print("Import summary:", counters)

    driver.close()
