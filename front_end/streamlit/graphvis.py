import streamlit as st

# def import_ontology():
#     from neo4j import GraphDatabase
    
#     url = "bolt://localhost:7687"
#     username = "neo4j"
#     password= "neo4j"
#     driver = GraphDatabase.driver(url, auth=(username, password))
   
#     def queryNeo4j(driver, query):
#         """Runs a single Cypher query."""
    
#         with driver.session() as session:
#             try:
#                 session.run(query)
                
#             except Exception as e:
#                 print(f"Error executing query: {e}")
#     queryConfig= """
#     MATCH(n) DETACH DELETE n
    
#     DROP CONSTRAINT n10s_unique_uri
    
#     CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;
    
#     CALL n10s.graphconfig.init({
#     handleVocabUris: "SHORTEN",
#     keepLangTag: false,
#     handleMultival: "ARRAY"})
#     """
    
#     importQuery = "CALL n10s.rdf.import.fetch(\"file:///home/lukas/CS425/Automatic_Taxonomy_Construction_NNs/data/owl/annett-o.owl\", \"RDF/XML\");"
#     # queryNeo4j()
        


def display_graph():
    voila_url = "http://localhost:8866"
    import os
    import time
    
    BASE_DIR = os.path.join(os.getcwd(), "data", "userowl")

    def list_files(startpath):
        """List only the .owl files inside the directory."""
        tree_lines = []
        for root, dirs, files in os.walk(startpath):
            for f in files:
                if f.endswith(".owl"):  # Only list .owl files
                    tree_lines.append(f"📄 {f}")
        return tree_lines

    @st.cache_data(ttl=10)  # Cache with TTL of 10 seconds
    def get_file_tree():
        """Get the current file tree. It will refresh every 10 seconds."""
        return list_files(BASE_DIR)

    # Streamlit UI
    st.title("🗂️ File Structure Viewer")
    
    if os.path.isdir(BASE_DIR):
        # Create a collapsible section for "Ontology files"
        with st.expander("🗂️ Ontology files"):
            file_tree = get_file_tree()
            if file_tree:
                for line in file_tree:
                    st.text(line)
            else:
                st.text("No .owl files found.")
    else:
        st.error(f"❌ Directory `{BASE_DIR}` not found. Please check the path.")
    
    iframe_html = f"""
        <style>
            body {{
                margin: 0;
                padding: 0;
            }}
            iframe {{
                width: 2700px;
                height: 100vh;
                border: none;
            }}
        </style>
        <iframe src="{voila_url}"></iframe>
    """

    # Embed the iframe in Streamlit
    st.markdown(iframe_html, unsafe_allow_html=True)
    


