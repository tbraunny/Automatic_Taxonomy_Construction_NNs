import streamlit as st
import streamlit.components.v1 as components

def display_graph():
    st.title("Graph Visualization & File Upload")
    import zipfile
    import os
    import io

    # Get the neural network architecture input from the user
    nn_architecture = st.text_input("Enter the name of the neural network architecture (e.g., alexnet):")

    # Display the user's input
    if nn_architecture:
        st.write(f"You entered: {nn_architecture}")
        nn_architecture = nn_architecture.lower()
    else:
        st.write("Please enter a neural network architecture.")

    # Base path to save the files
    base_path = "/home/lukas/CS425/Automatic_Taxonomy_Construction_NNs/data"

    # Function to ensure the directory exists for the architecture
    def ensure_directory(nn_architecture):
        directory_path = os.path.join(base_path, nn_architecture)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

    # Function to handle Python files
    def handle_python(uploaded_file, save_path):
        st.write(f"Processing Python file: {uploaded_file.name}")
        python_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        st.code(python_content, language='python')
        
        # Save the file to the designated folder
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success("Python file saved!")

    # Function to handle PDF files
    def handle_pdf(uploaded_file, save_path):
        # Example: Display the first 500 characters of the PDF content
        
        # Save the PDF file to the designated folder
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success("PDF file saved!")

    # Streamlit app layout
    st.title("File Upload Example")

    # Create a form for file upload
    with st.form(key='file_upload_form'):
        uploaded_file = st.file_uploader("Choose a file", type=["py", "pdf"])
        # uploaded_file = st.file_uploader("Choose a file", type=["zip", "py", "pdf"]) for zip implementation later
        submit_button = st.form_submit_button(label='Submit')

    # Handle file processing upon form submission
    if submit_button and uploaded_file is not None:
        if nn_architecture:  # Ensure that the user has entered an architecture name
            save_path = ensure_directory(nn_architecture)  # Get the folder path based on architecture name

            file_type = uploaded_file.type
            # if file_type == "application/zip":
            #     try:
            #         # Open the uploaded zip file
            #         with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            #             # Iterate through each file in the zip archive
            #             for file_name in zip_ref.namelist():
            #                 # Check if the file is a Python file
            #                 if file_name.endswith('.py'):
            #                     with zip_ref.open(file_name) as file:
            #                         handle_python(file, save_path)
            #                 elif file_name.endswith('.pdf'):
            #                     with zip_ref.open(file_name) as file:
            #                         handle_pdf(file, save_path)
            #                 else:
            #                     st.write(f"Skipping unsupported file: {file_name}")
            #     except zipfile.BadZipFile:
            #         st.error("The uploaded file is not a valid zip file.")
            #     except Exception as e:
            #         st.error(f"An error occurred while processing the zip file: {e}")
            if file_type == "text/x-python":
                handle_python(uploaded_file, save_path)
            elif file_type == "application/pdf":
                handle_pdf(uploaded_file, save_path)
            else:
                st.write("Unsupported file type.")
        else:
            st.error("Please enter a neural network architecture before uploading files.")
    
    
    animation_html = """
    <a href="http://100.102.166.78:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """

    st.markdown("Click below to view the Graph!")
    components.html(animation_html, height=400)
    

# def display_graphtest():
#     from py2neo import Graph
#     from pyvis.network import Network
#     import streamlit as st

#     st.title("Neo4j Graph Visualization")

#     st.sidebar.header("Login")
#     neo4j_uri = "bolt://localhost:7687"
#     neo4j_user = st.sidebar.text_input("Username", "neo4j")
#     neo4j_password = st.sidebar.text_input("Password", type="password")

#     if st.sidebar.button("Connect"):
#         try:
#             graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
#             st.session_state["graph"] = graph
#             st.success("Connected to Neo4j successfully!")
#         except Exception as e:
#             st.error(f"Connection failed: {e}")
    
#     st.subheader("Run Custom Cypher Query")
#     query = st.text_area("Enter your Cypher query", "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25")

#     if st.button("Run Query"):
#         if "graph" in st.session_state:
#             try:
#                 graph = st.session_state["graph"]
#                 result = graph.run(query).data()

#                 if not result:
#                     st.warning("No data returned from query.")
#                 else:
#                     # Create pyvis network object
#                     net = Network(height="600px", width="100%", notebook=True)

#                     # Add nodes and edges to the network
#                     for row in result:
#                         node1 = row["n"]
#                         node2 = row["m"]
#                         rel = row["r"]

#                         # Extract node IDs and labels
#                         id1 = node1.identity
#                         id2 = node2.identity
#                         label1 = list(node1.labels)[0] if node1.labels else "Node"
#                         label2 = list(node2.labels)[0] if node2.labels else "Node"

#                         # Add nodes
#                         net.add_node(id1, label=label1, title=str(node1))
#                         net.add_node(id2, label=label2, title=str(node2))

#                         # Add edges
#                         net.add_edge(id1, id2, label=rel["type"])

#                     # Generate and display the interactive network
#                     net.show("graph.html")
#                     st.components.v1.html(open("graph.html", "r").read(), height=600)

#             except Exception as e:
#                 st.error(f"Query execution failed: {e}")
#         else:
#             st.warning("Connect to Neo4j first!")