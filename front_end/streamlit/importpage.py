import streamlit as st
import streamlit.components.v1 as components
from utils.join_unique_dir_util import join_unique_dir
# from src.main import main
import time
import os
import shutil

def ontology_import():
    st.title("Graph Visualization & File Upload")

    user_ann_name = st.text_input("Enter the name of the neural network architecture (e.g., alexnet):")
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

    def handle_file(uploaded_file, save_path):
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        st.success(f"{uploaded_file.name} saved!")

    st.title("File Upload Example")

    with st.form(key='file_upload_form'):
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=["py", "pdf", "pb", "onnx"], 
            accept_multiple_files=True
        )
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and uploaded_files:
        if user_ann_name:
            ann_path = ensure_directory(user_ann_name)

            for uploaded_file in uploaded_files:
                file_type = uploaded_file.name.split('.')[-1].lower()

                if file_type in ["py", "pdf", "pb", "onnx"]:
                    handle_file(uploaded_file, ann_path)
                else:
                    st.warning(f"Unsupported file type: {uploaded_file.name}")
            
            user_owl_output = os.path.join(ann_path, user_ann_name + ".owl")
            # main(user_ann_name, ann_path, user_owl_output)
        else:
            st.error("Please enter a neural network architecture before uploading files.")

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

                            col1, col2, col3 = st.columns([1, 4, 1])

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
    
    animation_html = """
    <a href="http://localhost:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """
    st.markdown("Click below to view the Graph!")
    components.html(animation_html, height=400)
