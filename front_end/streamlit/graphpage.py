import streamlit as st
import streamlit.components.v1 as components
from utils.join_unique_dir_util import join_unique_dir
from src.main import main
import glob

def ontology_import():
    
    
    st.title("Graph Visualization & File Upload")
    # import zipfile
    import os
    
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

    # Streamlit app layout
    st.title("File Upload Example")

    # Create a form for file upload
    with st.form(key='file_upload_form'):
        uploaded_file = st.file_uploader("Choose a file", type=["py", "pdf"])
        # uploaded_file = st.file_uploader("Choose a file", type=["zip", "py", "pdf"]) for zip implementation later
        submit_button = st.form_submit_button(label='Submit')

    # Handle file processing upon form submission
    if submit_button and uploaded_file is not None:
        if user_ann_name:  # Ensure that the user has entered an architecture name
            ann_path = join_unique_dir(user_data_dir, user_ann_name)  # Get the folder path based on architecture name

            file_type = uploaded_file.type
            #this is code for zip for future reference
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
            if file_type == "application/octet-stream":
                handle_python(uploaded_file, ann_path)
            elif file_type == "application/pdf":
                handle_pdf(uploaded_file, ann_path)
            else:
                st.write("Unsupported file type.")
                
            
            
            
        
            user_owl_output = os.path.join(ann_path, user_ann_name + ".owl")
            
            main(user_ann_name, user_data_dir, user_owl_output)
            
            # main(user_ann_name,user_data_dir, user_owl_output)
        
        
        else:
            st.error("Please enter a neural network architecture before uploading files.")
            
        
    
    animation_html = """
    <a href="http://localhost:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """
    
    st.markdown("Click below to view the Graph!")
    components.html(animation_html, height=400)
    

