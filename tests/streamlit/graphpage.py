import streamlit as st

def display_graph():
    from py2neo import Graph
    from pyvis.network import Network
    import streamlit as st

    st.title("Neo4j Graph Visualization")

    st.sidebar.header("Login")
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = st.sidebar.text_input("Username", "neo4j")
    neo4j_password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Connect"):
        try:
            graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            st.session_state["graph"] = graph
            st.success("Connected to Neo4j successfully!")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    
    st.subheader("Run Custom Cypher Query")
    query = st.text_area("Enter your Cypher query", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")

    if st.button("Run Query"):
        if "graph" in st.session_state:
            try:
                graph = st.session_state["graph"]
                result = graph.run(query).data()

                if not result:
                    st.warning("No data returned from query.")
                else:
                    # Create pyvis network object
                    net = Network(height="600px", width="100%", notebook=True)

                    # Add nodes and edges to the network
                    for row in result:
                        node1 = row["n"]
                        node2 = row["m"]
                        rel = row["r"]

                        # Extract node IDs and labels
                        id1 = node1.identity
                        id2 = node2.identity
                        label1 = list(node1.labels)[0] if node1.labels else "Node"
                        label2 = list(node2.labels)[0] if node2.labels else "Node"

                        # Add nodes
                        net.add_node(id1, label=label1, title=str(node1))
                        net.add_node(id2, label=label2, title=str(node2))

                        # Add edges
                        net.add_edge(id1, id2, label=rel["type"])

                    # Generate and display the interactive network
                    net.show("graph.html")
                    st.components.v1.html(open("graph.html", "r").read(), height=600)

            except Exception as e:
                st.error(f"Query execution failed: {e}")
        else:
            st.warning("Connect to Neo4j first!")
