import streamlit as st

def display_graph():
    voila_url = "http://localhost:8866"
    
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
    

    
