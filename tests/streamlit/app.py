import streamlit as st
from graphpage import display_graph
from LLMchat import chat_page


# Page 1: Home
def home_page():
    with st.container():
        st.title("About TaxonNeuro:")
        
        st.markdown("""
        We are a research project for the Computer Science & Engineering [capstone program](https://www.unr.edu/engineering/student-resources/innovation-day/get-involved) at University of Nevada, Reno.
                    
        We present the newest tool for classifying & visualizing neural networks. We can classify, detail, visualize & summarize all things machine learning.
        We utilize a hierarchical organization technique called a taxonomy to organize the structures of neural networks. Think of the tree of life, with all living
        things in a hierarchy, perfectly organized for understanding what comes from where, and what it may relate to. Our tool will provide this organization, but
        for neural networks. You may be wondering what kind of inputs are compatible with this tool. Poorly documented code? We can classify it. Lengthy academic
        papers? We got you covered. Just have weights & biases saved? You bet your buns we can do that too.
             """)
    
    with st.container():
        st.header('News & Updates')
        st.markdown('Feb 1st: Welcome to the news & updates for TaxonNeuro, the most current information on this project will be posted here.')
        
    with st.container():
        st.header('Contributors from Team 3 (TaxonNeuro):')
        st.markdown("""
        - [Thomas Braun](https://www.linkedin.com/in/thomas-r-braun/)
        - [Lukas Lac](https://www.linkedin.com/in/lukas-lac/)
        - [Josue Ochoa](https://www.linkedin.com/in/josuejochoa/)
        - [Rich White](https://www.linkedin.com/in/richardwhitein/)
        """)

    with st.container():
        st.header('Advisors:')
        st.markdown("""
        - [Chase Carthen (PhD Student of CSE at University of Nevada, Reno)](https://scholar.google.com/citations?user=X9yIPe4AAAAJ&hl=en)
        - [Dr. Alireza Tavakkoli (Associate Professor of CSE at University of Nevada, Reno)](https://www.unr.edu/cse/people/alireza-tavakkoli)
        """)
    with st.container():
        st.header('Instructors:')
        st.markdown("""
        - [Sara Davis]
        - [David Feil-Seifer]
        - [Vinh Le]
        - [Levi Scully]
        """)


# Sidebar for navigation
page = st.sidebar.radio("Select a Page", ("Home", "Chat with AI", "Graph"))

# Conditional rendering of pages based on selection
if page == "Home":
    home_page()
elif page == "Chat with AI":
    chat_page()
elif page == "Graph":
    display_graph()