import streamlit as st
from importpage import import_page
from LLMchat import chat_page
from graphvis import display_graph
from mcpconnection import mcp_connector
import asyncio
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        .stAppDeployButton {display: none;}
        
    </style>
    """,
    unsafe_allow_html=True
)
# Page 1: Home
def home_page():

    try:
        from PIL import Image
        import base64
        from io import BytesIO

        with st.container():
            # HTML and Markdown for the banner with logo on the left
            st.markdown(f"""
                <div style='display: flex; align-items: center; justify-content: center; padding: 30px; background-color: #ffecb3; border-radius: 10px; border: 1px solid #fb8c00; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);'>
                    <div style="text-align: center;">
                        <h1 style='font-family: "Arial", sans-serif; color: #fb8c00; margin: 0; font-size: 36px;'>🧠 About <span style='color:#ef6c00;'>TaxonNeuro</span></h1>
                        <p style='font-family: "Arial", sans-serif; font-size: 18px; color: #6d4c41; margin-top: 5px;'>Automatic Taxonomy Construction for Neural Networks</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # PROJECT DESCRIPTION
            st.markdown("""
            <div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.7; color: #333; padding: 20px;">
            <p>
                We are a research project for the Computer Science & Engineering 
                <a href="https://www.unr.edu/engineering/student-resources/innovation-day/get-involved" 
                target="_blank" style="color: #fb8c00; font-weight: bold;">
                capstone program
                </a> 
                at the University of Nevada, Reno.
            </p>

            <p>
                <strong>TaxonNeuro</strong> is your new best friend when it comes to classifying and visualizing neural networks. 
                We classify, detail, visualize, and summarize all things machine learning.
            </p>

            <p>
                Our core methodology revolves around hierarchical organization — a technique called a 
                <strong>taxonomy</strong>. Think of it like the tree of life for neural networks: a clean, logical breakdown 
                of models and architectures, where everything has a place and a purpose.
            </p>

            <p>
                Got poorly documented code? We’ll classify it. Reading a dense academic paper? We’ve got you covered. 
                Just have saved weights & biases? You bet your buns we can work with that too.
            </p>
            </div>
            """, unsafe_allow_html=True)

            # NEWS & UPDATES
            with st.container():
                st.markdown("""
                <h2 style="color:#ef6c00;">📰 News & Updates</h2>
                <ul style="font-family: Arial, sans-serif; font-size: 16px; color: #444;">
                    <li><strong>Feb 1st:</strong> Welcome to the TaxonNeuro News Hub! Stay tuned for the latest features, bug fixes, and breakthroughs.</li>
                </ul>
                """, unsafe_allow_html=True)

            # CONTRIBUTORS
            with st.container():
                st.markdown("""
                <h2 style="color:#ef6c00;">🤝 Contributors from Team 3 (TaxonNeuro)</h2>
                <ul style="font-family: Arial, sans-serif; font-size: 16px; color: #444;">
                    <li><a href="https://www.linkedin.com/in/thomas-r-braun/" target="_blank">Thomas Braun</a></li>
                    <li><a href="https://www.linkedin.com/in/lukas-lac/" target="_blank">Lukas Lac</a></li>
                    <li><a href="https://www.linkedin.com/in/josuejochoa/" target="_blank">Josue Ochoa</a></li>
                    <li><a href="https://www.linkedin.com/in/richardwhitein/" target="_blank">Rich White</a></li>
                </ul>
                """, unsafe_allow_html=True)

            # ADVISORS
            with st.container():
                st.markdown("""
                <h2 style="color:#ef6c00;">📚 Advisors</h2>
                <ul style="font-family: Arial, sans-serif; font-size: 16px; color: #444;">
                    <li><a href="https://scholar.google.com/citations?user=X9yIPe4AAAAJ&hl=en" target="_blank">Chase Carthen</a> (PhD Student of CSE at UNR)</li>
                    <li><a href="https://www.unr.edu/cse/people/alireza-tavakkoli" target="_blank">Dr. Alireza Tavakkoli</a> (Associate Professor, UNR)</li>
                    <li><a href="https://www.unr.edu/cse/people/fred-harris" target="_blank">Dr. Fred Harris, Jr.</a> (Associate Dean, UNR)</li>
                </ul>
                """, unsafe_allow_html=True)

            # INSTRUCTORS
            with st.container():
                st.markdown("<h2 style='color:#ef6c00;'>👩‍🏫 Instructors</h2>", unsafe_allow_html=True)
                with st.expander("List of Instructors"):
                    st.markdown("""
                    - Sara Davis  
                    - David Feil-Seifer  
                    - Vinh Le  
                    - Levi Scully  
                    """)

            # RESOURCES
            with st.container():
                st.markdown("<h2 style='color:#ef6c00;'>📂 Project-Related Resources</h2>", unsafe_allow_html=True)
                with st.expander("List of Resources"):
                    st.markdown("""
                    - [Ontology Population Using LLM's](https://arxiv.org/abs/2411.01612)  
                    - [Semantic Similarity of Ontology Instances Tailored on the Application Context](https://www.researchgate.net/publication/220830109_Semantic_Similarity_of_Ontology_Instances_Tailored_on_the_Application_Context)  
                    - [Making Neural Networks FAIR](https://arxiv.org/pdf/1907.11569#page=1.50)  
                    - [deepseek-r1](https://ollama.com/library/deepseek-r1)  
                    - [ollama](https://ollama.com/)  
                    """)

        with st.container():
            # Load the image and convert it to base64
            image_path = 'front_end/streamlit/data/images/Taxon_Neuro_image.png'
            image = Image.open(image_path)

            # Convert the image to base64 so it can be embedded in the HTML
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Display the link and image inside the text
            st.markdown("""
            <div style="text-align: left; font-family: Arial, sans-serif; margin-top: 10px;">
                <p style="color: #fb8c00; font-size: 18px; font-weight: bold;">
                    🔗 Click the logo below to check out our GitHub Repository!
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <a href="https://github.com/tbraunny/Automatic_Taxonomy_Construction_NNs/" target="_blank">
                <img src="data:image/png;base64,{img_str}" 
                    style="width: 221px; height: 124px; vertical-align: middle; margin-left: 10px;
                            border-radius: 10px; 
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
                            transition: transform 0.3s ease-in-out;"/>
            </a>
            <style>
                a:hover img {{
                    transform: scale(1.05); /* Slight zoom effect on hover */
                }}
            </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An unexpected error occurred. Please try again later. 🚨")
try:
    # Sidebar for navigation
    st.sidebar.markdown("<h2 style='font-family: Arial, sans-serif; color: #fb8c00;'>Welcome to TaxonNeuro!</h2>", unsafe_allow_html=True)

    page = st.sidebar.selectbox("Choose an Option", ("🏠 Home", "🤖 Chat with AI", "📥 Import", "📊 Graph", "Test"))

    # Conditional rendering of pages based on selection
    if page == "🏠 Home":
        home_page()
    elif page == "🤖 Chat with AI":
        chat_page()
    elif page == "📥 Import":
        import_page()
    elif page == "📊 Graph":
        display_graph()
    elif page == "Test":
        asyncio.run( mcp_connector())
    
except Exception as e:
        st.error(f"An unexpected error occurred. Please try again later. 🚨")