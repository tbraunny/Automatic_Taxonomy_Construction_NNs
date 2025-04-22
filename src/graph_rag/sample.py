from neo4j import GraphDatabase

# Connect to your Neo4j instance
def get_neo4j_connection():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "neo4j"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver

def import_rdf(session, url,  rdf_format="RDF/XML", options=None, inline=False, file_contents=""):
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
    
    if inline and file_contents != None:
        cypher_query = """
        CALL n10s.rdf.import.inline($file_contents, 'RDF/XML')
        """


    init_query = """
    CALL n10s.graphconfig.init($options)
    """
    setup_query = """
    CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

    """
    # CALL n10s.graphconfig.init()
    drop = session.run(deleteall, url=url)
    drop = session.run(dropconstraint, url=url)
    drop = session.run(createconstraint, url=url)
    drop = session.run(graphconfiginit, url=url)
    result = session.run(cypher_query, file_contents=file_contents,url=url, format=rdf_format, options=options)
    summary = result.consume()
    return summary.counters

if __name__ == '__main__':
    driver = neo4j_connection()
    with driver.session() as session:
        file_path = "file:///import/annett-o-0.1.owl"
        rdf_format = "RDF/XML"
        options = {
            "handleVocabUris": "SHORTEN",
            "typesToLabels": False,
            "keepLangTag": False,
            "handleMultival": "ARRAY"
        }
        
        content = open('src/graph_rag/import/annett-o-0.1.owl','r').read()
        counters = import_rdf(session, file_path, rdf_format, options,inline=True,file_contents=content)
        print("Import summary:", counters)

    driver.close()
