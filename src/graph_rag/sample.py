from neo4j import GraphDatabase

# Connect to your Neo4j instance
uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j"

driver = GraphDatabase.driver(uri, auth=(username, password))

def import_rdf(session, url, rdf_format="RDF/XML", options=None):
    if options is None:
        options = {}

    deleteall = """
    MATCH(n) DETACH DELETE n;
    """
    dropconstraint = """
    DROP CONSTRAINT n10s_unique_uri;
    """
    
    createconstraint = """
    CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
    """

    graphconfiginit = """
    CALL n10s.graphconfig.init({
        handleVocabUris: "SHORTEN",
        keepLangTag: false,
        handleMultival: "ARRAY"
    });
    """

    cypher_query = """
    CALL n10s.rdf.import.fetch($url, $format, $options)
    """
    init_query = """
    CALL n10s.graphconfig.init()
    """
    setup_query = """
    CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

    """
    # CALL n10s.graphconfig.init()
    drop = session.run(deleteall, url=url)
    drop = session.run(dropconstraint, url=url)
    drop = session.run(createconstraint, url=url)
    drop = session.run(graphconfiginit, url=url)
    result = session.run(cypher_query, url=url, format=rdf_format, options=options)
    summary = result.consume()
    return summary.counters

if __name__ == '__main__':
    with driver.session() as session:
        file_path = "file:///import/annett-o-0.1.owl"
        rdf_format = "RDF/XML"
        options = {
            "handleVocabUris": "SHORTEN",
            "typesToLabels": False,
            "keepLangTag": False,
            "handleMultival": "ARRAY"
        }

        counters = import_rdf(session, file_path, rdf_format, options)
        print("Import summary:", counters)

    driver.close()
