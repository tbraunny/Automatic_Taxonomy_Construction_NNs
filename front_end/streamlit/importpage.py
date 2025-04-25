import streamlit as st
import streamlit.components.v1 as components
from utils.join_unique_dir_util import join_unique_dir
from src.main import main, remove_ann_config_from_user_owl
import time
import os
import shutil
from utils.exception_utils import *

from utils.logger_util import get_logger
logger = get_logger("importpage", max_logs=3)

# A global list to store success message placeholders
success_placeholders = []
st.session_state.first_import = True

def show_error(e):
    logger.error(f"Custom exception occurred", exc_info=True)
    error_data = e.to_dict() if hasattr(e, "to_dict") else {"message": "An unexpected error occurred during the process. Please try again."}
    st.error(f"{error_data.get('message')}")
    # if error_data.get("context"):
    #     st.caption(f"Context: {error_data['context']}")
    # if error_data.get("code"):
    #     st.caption(f"Error Code: {error_data['code']}")


def import_ontology_to_neo4j():
    from neo4j import GraphDatabase

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "../../data/userinput/user_owl.owl")
    file_path = os.path.normpath(file_path)
    
    url = "bolt://0.0.0.0:7687"
    username = "neo4j"
    password = "neo4j"
    driver = GraphDatabase.driver(url, auth=(username, password))

    def queryNeo4j(driver, query):
        """Runs a single Cypher query."""
        with driver.session() as session:
            try:
                session.run(query)
            except Exception as e:
                print(f"Error executing query: {e}")

    importQuery = f'CALL n10s.rdf.import.fetch("file://{file_path}", "RDF/XML");'
    queryNeo4j(driver, "MATCH(n) DETACH DELETE n;")
    queryNeo4j(driver, "DROP CONSTRAINT n10s_unique_uri;")
    queryNeo4j(driver, "CREATE CONSTRAINT n10s_unique_uri FOR (r:Resource) REQUIRE r.uri IS UNIQUE;")
    queryNeo4j(driver, "CALL n10s.graphconfig.init({handleVocabUris: \"SHORTEN\", keepLangTag: false, handleMultival: \"ARRAY\"});")
    queryNeo4j(driver, importQuery)
    queryNeo4j(driver, """
    MATCH (n)
    WHERE n.uri STARTS WITH 'http://w3id.org/annett-o/'
    SET n.uri = SPLIT(n.uri, '/')[SIZE(SPLIT(n.uri, '/')) - 1]
    """)

def import_page():
    st.title("Graph Visualization & File Upload")

    user_ann_name = st.text_input("Enter the name of the neural network architecture (e.g., alexnet):")
    if user_ann_name:
        st.write(f"You entered: {user_ann_name}")
        user_ann_name = user_ann_name.lower()
    else:
        st.write("Please enter a neural network architecture.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_dir = os.path.join(script_dir, "../../data/userinput")

    def handle_file(uploaded_file, save_path):
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        success_placeholder = st.empty()
        success_placeholder.success(f"{uploaded_file.name} saved!")
        time.sleep(2)
        success_placeholder.empty()

    st.title("File Upload Example")

    with st.form(key='file_upload_form'):
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["py", "pdf", "pb", "onnx"],
            accept_multiple_files=True
        )
        submit_button = st.form_submit_button(label='Submit')

    # Check if `main` is already running
    if "is_main_running" not in st.session_state:
        st.session_state.is_main_running = False

    # Track whether the process is running
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False

    if submit_button and uploaded_files:
        if not user_ann_name:
            st.error("Please enter a neural network architecture before uploading files.")
        else:
            ann_path = os.path.join(user_data_dir, user_ann_name)

            if os.path.exists(ann_path):
                st.warning(f"Files for architecture '{user_ann_name}' already exist. Please choose a new architecture name")
            else:
                os.makedirs(ann_path)

                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.name.split('.')[-1].lower()

                    if file_type in ["py", "pdf", "pb", "onnx"]:
                        handle_file(uploaded_file, ann_path)
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")
                
                # Prevent running `main` again if it's already in progress
                if st.session_state.is_main_running:
                    st.warning("The processing is already running. Please wait until it's finished.")
                else:
                    # Mark the process as running in session state
                    st.session_state.is_main_running = True
                    st.session_state.is_processing = True
                    
                    # Show spinner
                    with st.spinner("Processing your files and generating ontology. Please wait..."):
                        try:
                            # Run the main function
                            main(user_ann_name, ann_path, use_user_owl=True)
                        except (CodeExtractionError, PDFError, LLMAPIError, DatabaseError) as e:
                            show_error(e)
                        except Exception as e:
                            logger.error("An unexpected error occurred in main:\n\n", exc_info=True)
                            st.error(f"An unexpected error occurred during the process. Please try again.")
                        finally:
                            # Mark that `main` has finished running
                            st.session_state.is_main_running = False
                            st.session_state.is_processing = False
                    
                            # Once processing is done, display info and import ontology to Neo4j
                            info_placeholder = st.empty()
                            info_placeholder.info("Finished processing files. Now importing ontology into Neo4j...")
                            import_ontology_to_neo4j()
                            info_placeholder.empty()
                            st.success("Ontology successfully imported into Neo4j!")

                st.header("View Uploaded Files")

    if os.path.exists(user_data_dir):
        arch_dirs = sorted([d for d in os.listdir(user_data_dir) if os.path.isdir(os.path.join(user_data_dir, d))])
        if arch_dirs:
            for arch in arch_dirs:
                arch_path = os.path.join(user_data_dir, arch)

                with st.expander(f"📁 {arch}"):
                    files = os.listdir(arch_path)
                    if files:
                        for file in files:
                            file_path = os.path.join(arch_path, file)
                            file_mod_time = time.ctime(os.path.getmtime(file_path))

                            col1, col2, col3, col4 = st.columns([1, 3, 1, 2])

                            with col1:
                                st.markdown(f"- **{file}**")
                                st.caption(f"Uploaded: {file_mod_time}")

                            with col3:
                                if st.button(f"🗑️ Delete {file}", key=f"delete_{arch}_{file}"):
                                    os.remove(file_path)
                                    st.success(f"Deleted `{file}` from `{arch}`")
                                    st.rerun()
                    else:
                        st.write("No files uploaded yet.")

                    if st.button(f"🧹 Delete entire `{arch}` folder", key=f"delete_folder_{arch}"):
                        shutil.rmtree(arch_path)
                        st.success(f"Deleted entire architecture folder: `{arch}`")
                        st.rerun()
        else:
            st.info("No architectures found yet.")
    else:
        st.warning("User input directory does not exist.")

    # animation_html = """
    # <a href="http://100.102.166.78:8866/" target="_blank">
    #     <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
    #     <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    # </a>
    # """
    animation_html = """
    <a href="http://172.24.218.133:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """
    st.markdown("Click below to view the graph in a new tab!")
    components.html(animation_html, height=400)
