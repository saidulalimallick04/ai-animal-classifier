import streamlit as st

st.set_page_config(
    page_title="AI Animal Classifier", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def home():
    st.title("Home")

    # First Section
    st.subheader("Welcome to the AI Animal Classifier")
    st.markdown("---")  

    # Second Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("How it works")
        st.markdown(
            """
            1. Upload an image of an animal.
            2. The app will use the pre-trained model to classify the animal.
            3. The app will display the results.
            """
        )
    with col2:
        st.image(
            "assets/sample-1.png",
            caption="AI Animal Classifier",
            width='stretch',
        )
    st.markdown("---")  



    # Last Section
    # Two‑column layout: image on the left, features on the right
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1764712755002-ce3400921a7c?q=80&w=1119&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            caption="AI Animal Classifier",
            width='stretch',
        )
    with col2:
        st.subheader("Features")
        st.markdown(
            """
- **Multi-page Streamlit app** with dashboard, live prediction, history, and model details.
- **Model selector** in the sidebar to choose `.h5`/`.keras` models.
- **Image preprocessing** and classification for 99 animal classes.
- **Rich visualizations** of dataset statistics and model performance.
- **Extensible architecture** for adding new models and datasets.
"""
        )
    st.markdown("---")
pages = [
    st.Page(home, title="Home", icon="🏠", url_path="/",default=True),
    st.Page("pages/01_Dashboard_Page.py", title="Dashboard", icon="📊", url_path="dashboard/"),
    st.Page("pages/02_Live_Prediction_Page.py", title="Live Prediction", icon="🎥", url_path="live-prediction/"),
    st.Page("pages/05_Prediction_History_Page.py", title="Prediction History", icon="📜", url_path="prediction-history/"),
    st.Page("pages/06_Model_Details_Page.py", title="Model Details", icon="🛠️", url_path="model-details/"),
    st.Page("pages/09_About_Page.py", title="About", icon="ℹ️", url_path="about/"),
    st.Page("pages/10_Old_App.py", title="Old Interface", icon="🔙", url_path="old-interface/"),
]

pg = st.navigation(pages)
pg.run()
