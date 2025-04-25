import streamlit as st

def display_graph():
    try:
        voila_url = "http://100.102.166.78:8866" # for tailscale
        # voila_url = "http://172.24.218.133:8866"
        
        # Ontology Visualization Section
        st.markdown(
            "<h1 style='font-family: Arial, sans-serif; color: #fb8c00;'>Ontology Visualization</h1>", 
            unsafe_allow_html=True
        )

        iframe_html = f"""
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                }}
                iframe {{
                    width: 100%;
                    height: 1120px; 
                    border: none;
                }}
            </style>
            <iframe src="{voila_url}"></iframe>
        """

        # Embed the iframe for the graph
        st.markdown(iframe_html, unsafe_allow_html=True)

        # Add some space between the sections (optional)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Taxonomy Generator Section
        st.markdown(
            "<h1 style='font-family: Arial, sans-serif; color: #fb8c00;'>Taxonomy Generator </h1>", 
            unsafe_allow_html=True
        )

        # Create a description for the user
        st.markdown("""
            Click the button below to generate and display the taxonomy table.
            The table contains various aspects of taxonomy data and can help you understand the structure and organization.
        """)
        
        def display_taxonomy():
            import pandas as pd 
            import numpy as np

            # Create a button to generate the taxonomy
            if st.button('Generate Taxonomy'):
                with st.spinner('Loading... Please wait while we fetch the taxonomy data!'):
                    try:
                        # Attempt to read the CSV file
                        df = pd.read_csv('data/taxonomy/faceted/generic/generic_taxonomy.csv')
                        
                        # Show the DataFrame in a stylish way
                        st.success('Taxonomy data loaded successfully! 🎉')
                        st.table(df)  # Display the table

                    except FileNotFoundError as e:
                        st.error(f"Oops! The Taxonomy was not found. 😟")
                        st.exception(e)  # Show the exception for debugging
                    except pd.errors.EmptyDataError as e:
                        st.warning(f"Warning: The file is empty. 😔")
                        st.exception(e)  # Show the exception for debugging
                    except Exception as e:
                        st.error(f"An unexpected error occurred. Please try again later. 🚨")
                        st.exception(e)  # Show the exception for debugging
        
        display_taxonomy()
    
    except Exception as e:
        st.error(f"An unexpected error occurred. Please try again later. 🚨")