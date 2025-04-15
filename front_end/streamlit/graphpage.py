import streamlit as st
import streamlit.components.v1 as components
from utils.join_unique_dir_util import join_unique_dir
from src.main import main
import time
import os
import shutil
def ontology_import():
    st.title("Graph Visualization & File Upload")
    # Get the neural network architecture input from the user
    user_ann_name = st.text_input("Enter the name of the neural network architecture (e.g., alexnet):")

    # Display the user's input
    if user_ann_name:
        st.write(f"You entered: {user_ann_name}")
        user_ann_name = user_ann_name.lower()
    else:
        st.write("Please enter a neural network architecture.")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    user_data_dir = os.path.join(script_dir, "../../data/userinput")

    def ensure_directory(user_ann_name):
        directory_path = os.path.join(user_data_dir, user_ann_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

    def handle_python(uploaded_file, save_path):
        python_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
        # Save the file to the designated folder
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success("Python file saved!")

    def handle_pdf(uploaded_file, save_path):
        # Save the PDF file to the designated folder
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success("PDF file saved!")
        st.markdown("---")
        
    # Streamlit app layout
    st.title("File Upload Example")

    # Create a form for file upload
    with st.form(key='file_upload_form'):
        uploaded_files = st.file_uploader("Choose a file", type=["py", "pdf"], accept_multiple_files=True)
        # uploaded_file = st.file_uploader("Choose a file", type=["zip", "py", "pdf"]) for zip implementation later
        submit_button = st.form_submit_button(label='Submit')

    # Handle file processing upon form submission
    if submit_button and uploaded_files:
        
        if user_ann_name:
            ann_path = ensure_directory(user_ann_name)

            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type

                if file_type == "application/octet-stream":
                    handle_python(uploaded_file, ann_path)
                elif file_type == "application/pdf":
                    handle_pdf(uploaded_file, ann_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
            
            user_owl_output = os.path.join(ann_path, user_ann_name + ".owl")
            # main(user_ann_name, ann_path, user_owl_output)
        else:
            st.error("Please enter a neural network architecture before uploading files.")

                
            # user_owl_output = os.path.join(ann_path, user_ann_name + ".owl")
    
            
            # main(user_ann_name, ann_path, user_owl_output)

    #file view
    
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

                            col1, col2, col3 = st.columns([4, 3, 1])

                            with col1:
                                st.markdown(f"- **{file}**")
                                st.caption(f"Uploaded: {file_mod_time}")
                            # with col2:
                                # with open(file_path, "rb") as f:
                                #     st.download_button(
                                #         label="Download",
                                #         data=f,
                                #         file_name=file,
                                #         mime="application/octet-stream",
                                #         key=f"download_{arch}_{file}"  # Unique key per file
                                #     )
                            with col3:
                                if st.button(f"🗑️ Delete {file}", key=f"delete_{arch}_{file}"):
                                    os.remove(file_path)
                                    st.success(f"Deleted `{file}` from `{arch}`")
                                    st.rerun()  # Refresh the page to reflect the change
                    else:
                        st.write("No files uploaded yet.")

                    # Option to delete the entire architecture folder
                    if st.button(f"🧹 Delete entire `{arch}` folder", key=f"delete_folder_{arch}"):
                        shutil.rmtree(arch_path)
                        st.success(f"Deleted entire architecture folder: `{arch}`")
                        st.rerun()  # Refresh the page to reflect the change
        else:
            st.info("No architectures found yet.")
    else:
        st.warning("User input directory does not exist.")
    
    animation_html = """
    <a href="http://localhost:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """
    
    st.markdown("Click below to view the Graph!")
    components.html(animation_html, height=400)
    

