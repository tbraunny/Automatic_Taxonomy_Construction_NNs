import streamlit as st

# Embedding the Voila-hosted notebook as an iframe
st.title("Voila Embedded in Streamlit")
st.markdown("[Thomas Braun](http://localhost:8866/)")
# Adjust the URL to match where Voila is running (e.g., localhost:8866)
voila_url = "http://127.0.0.1:8866/"

# Embed the notebook
iframe_code = f'<iframe src="{voila_url}" width="100%" height="800"></iframe>'
st.markdown(iframe_code, unsafe_allow_html=True)



