import streamlit as st

def display_graph():
    try:
        #voila_url = "http://100.102.166.78:8866" # for tailscale
        # voila_url = "http://172.24.218.133:8866"
        voila_url = "http://localhost:8866"
    
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

    except Exception as e:
        st.error(f"An unexpected error occurred. Please try again later. 🚨")