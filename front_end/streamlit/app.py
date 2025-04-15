import streamlit as st
from graphpage import ontology_import
from LLMchat import chat_page
from graphvis import display_graph

st.set_page_config(layout="wide")

# Page 1: Home
def home_page():
    with st.container():
        st.title("About TaxonNeuro:")
        
        st.markdown("""
        We are a research project for the Computer Science & Engineering [capstone program](https://www.unr.edu/engineering/student-resources/innovation-day/get-involved) at University of Nevada, Reno.
                    
        We present the newest tool for classifying & visualizing neural networks. We can classify, detail, visualize & summarize all things machine learning.
        We utilize a hierarchical organization technique called a taxonomy to organize the structures of neural networks. 
        <br> Think of the tree of life, with all living things in a hierarchy, perfectly organized for understanding what comes from where, and what it may relate to. 
        <br> Our tool will provide this organization, but for neural networks. You may be wondering what kind of inputs are compatible with this tool. Poorly documented code? We can classify it. Lengthy academic
        papers? We got you covered. Just have weights & biases saved? You bet your buns we can do that too.
             """, unsafe_allow_html=True)
    
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
        - [Dr. Fred Harris, Jr. (Associate Dean of Faculty and Academic Affairs; Foundation Professor of Computer Science & Engineering)](https://www.unr.edu/cse/people/fred-harris)
        """)
    with st.container():
        st.header('Instructors:')
        with st.expander("list of Instructors"):
            st.markdown("""
            - [Sara Davis]
            - [David Feil-Seifer]
            - [Vinh Le]
            - [Levi Scully]
            """)
        
    with st.container():
        st.header('Project-Related Resources')
        
        with st.expander("list of resources"):
            st.markdown("""
            - [Ontology Population Using LLM's](https://arxiv.org/abs/2411.01612)
            - [Semantic Similarity of Ontology Instances Tailored on the Application Context](https://www.researchgate.net/publication/220830109_Semantic_Similarity_of_Ontology_Instances_Tailored_on_the_Application_Context)
            - [Making Neural Networks FAIR](https://arxiv.org/pdf/1907.11569#page=1.50)
            - [deepseek-r1](https://ollama.com/library/deepseek-r1)
            - [ollama](https://ollama.com/)
            """)
    
    with st.container():
        st.header('Check out our [GitHub repository](https://github.com/tbraunny/Automatic_Taxonomy_Construction_NNs/)!')

# Sidebar for navigation
st.sidebar.markdown("## Welcome to TaxonNeuro!")

page = st.sidebar.selectbox("Choose an Option", ("🏠 Home", "🤖 Chat with AI", "📊 Import", "Graph"))

# Conditional rendering of pages based on selection
if page == "🏠 Home":
    home_page()
elif page == "🤖 Chat with AI":
    chat_page()
elif page == "📊 Import":
    ontology_import()
elif page == "Graph":
    display_graph()