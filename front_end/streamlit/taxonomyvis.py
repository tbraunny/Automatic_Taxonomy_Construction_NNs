import streamlit as st

def taxonomy_page():
    from src.taxonomy.custom_taxonomy import generate_custom_taxonomy
    try:
            # Taxonomy Generator Section
        st.markdown(
            "<h1 style='font-family: Arial, sans-serif; color: #fb8c00;'>Taxonomy Generator </h1>", 
            unsafe_allow_html=True
        )

        # Create a description for the user
        st.markdown("""
        <div style="font-size:25px;">
            Click the button below to generate and display the taxonomy table.<br>
            The table contains various aspects of taxonomy data and can help you understand the structure and organization.
        </div>
    """, unsafe_allow_html=True)
        
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
                        st.dataframe(df)  # Display the table

                    except FileNotFoundError as e:
                        st.error(f"Oops! The Taxonomy was not found. 😟")
                        st.exception(e)  # Show the exception for debugging
                    except pd.errors.EmptyDataError as e:
                        st.warning(f"Warning: The file is empty. 😔")
                        st.exception(e)  # Show the exception for debugging
                    except Exception as e:
                        st.error(f"An unexpected error occurred. Please try again later. 🚨")
                        st.exception(e)  # Show the exception for debugging
            st.markdown("""
        <div style="font-size:25px;">
            Here you can enter your own query to generate a custom taxonomy.
        </div>
    """, unsafe_allow_html=True)
            
            user_query = st.text_input("Enter your query for taxonomy:")
            if st.button('Generate Custom Taxonomy'):
                st.write(f"Generating taxonomy for: {user_query}")
                with st.spinner('Loading... Please wait while we fetch the taxonomy data!'):
                    try:
                        # Call the custom taxonomy generation function
                        generate_custom_taxonomy(user_query)
                        
                        # Attempt to read the CSV file
                        df = pd.read_csv('data/taxonomy/faceted/custom/user_custom_taxonomy.csv')
                        
                        # Show the DataFrame in a stylish way
                        st.success('Taxonomy data loaded successfully! 🎉')
                        st.dataframe(df)  # Display the table

                    except FileNotFoundError as e:
                        st.error(f"Oops! The Taxonomy was not found. 😟")
                        st.exception(e)
        
        display_taxonomy()
    except Exception as e:
        st.error(f"An unexpected error occurred. Please try again later. 🚨")