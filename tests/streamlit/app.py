import streamlit as st
from graphpage import display_graph
from LLMchat import chat_page
import webbrowser


# st.markdown(
#     """
#     <style>
#     .banner {
#         background-color: #4CAF50; 
#         color: white; 
#         font-size: 36px; 
#         padding: 20px;
#         text-align: center;
#         border-radius: 10px;
#     }
#     </style>
#     <div class="banner">
#         Welcome to My Streamlit App!
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

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

def test_page():
    import streamlit as st
    import streamlit.components.v1 as components
# Your main content here
    animation_html = """
    <a href="http://localhost:8866/" target="_blank">
        <script src="https://unpkg.com/@dotlottie/player-component@2.7.12/dist/dotlottie-player.mjs" type="module"></script>
        <dotlottie-player src="https://lottie.host/756ea83b-4c33-4d3a-a2ac-3fa9050f1c8f/j7jKHC8GEv.lottie" background="transparent" speed="1" style="width: 300px; height: 300px" loop autoplay></dotlottie-player>
    </a>
    """

    # Use Streamlit to display the animation as a clickable button
    st.title("Clickable Lottie Animation")
    components.html(animation_html, height=400)

# Sidebar for navigation
st.sidebar.markdown("## Welcome to TaxonNeuro!")

page = st.sidebar.selectbox("Choose an Option", ("🏠 Home", "🤖 Chat with AI", "📊 Graph", "🔬 Test"))

# Conditional rendering of pages based on selection
if page == "🏠 Home":
    home_page()
elif page == "🤖 Chat with AI":
    chat_page()
elif page == "📊 Graph":
    display_graph()
elif page == "🔬 Test":
    test_page()