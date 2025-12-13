# pages/09_About_Page.py
import streamlit as st

def run():
    st.title("ℹ️ About the Project")
    st.write("A deep dive into the **AI Animal Classifier** — from concept to code.")
    st.markdown("---")

    # 1. Project Overview
    st.header("🌟 Project Overview")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("""
        The **AI Animal Classifier** is a state-of-the-art computer vision application designed to identify 99 different animal species from images. 
        It leverages the power of Deep Learning, specifically Convolutional Neural Networks (CNNs), to analyze visual patterns and predict animal categories with high accuracy.
        
        Use cases include educational tools, wildlife monitoring assistance, or simply satisfying your curiosity about that strange bug in your garden!
        """)
    with c2:
        st.image("https://images.unsplash.com/photo-1555169062-013468b47731?q=80&w=1000&auto=format&fit=crop", caption="AI & Nature", width="stretch")

    st.markdown("---")

    # 2. Tech Stack
    st.header("🛠️ Technology Stack")
    st.write("We used a modern, robust stack to build this application:")
    
    mac1, mac2, mac3, mac4 = st.columns(4)
    with mac1:
        st.markdown("### 🧠 Core AI")
        st.markdown("- **TensorFlow & Keras**")
        st.markdown("- **CNN Architecture**")
        st.markdown("- **NumPy & Pandas**")
    
    with mac2:
        st.markdown("### 🎨 Frontend")
        st.markdown("- **Streamlit** (UI Framework)")
        st.markdown("- **Pillow** (Image Processing)")
        st.markdown("- **Matplotlib** (Visualization)")
        
    with mac3:
        st.markdown("### ⚙️ Backend")
        st.markdown("- **Python 3.10+**")
        st.markdown("- **Session State Management**")
        st.markdown("- **CSV Persistence**")
        
    with mac4:
        st.markdown("### 🔧 Tools")
        st.markdown("- **VS Code**")
        st.markdown("- **Git & GitHub**")
        st.markdown("- **Google Colab** (Training)")

    st.markdown("---")

    # 3. Development Journey
    st.header("🚀 How We Developed It")
    with st.expander("Step 1: Data Collection & Cleaning", expanded=True):
        st.write("We curated a dataset of 99 animal categories. Images were resized, normalized, and augmented (rotation, flips) to ensure the model generalizes well to new data.")
        
    with st.expander("Step 2: Model Training", expanded=True):
        st.write("We designed a custom CNN architecture with multiple Convolutional and MaxPooling layers. The model was trained using the Adam optimizer and Categorical Crossentropy loss function over 30+ epochs.")
        
    with st.expander("Step 3: Web App Integration", expanded=True):
        st.write("We built a multi-page interactive dashboard using Streamlit, allowing users to upload images, view real-time predictions, track history, and analyze model metadata.")

    st.markdown("---")

    # 4. How to Use
    st.header("📖 How to Use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**1. Dashboard**")
        st.write("Go to the Dashboard, upload an image (JPG/PNG), and convert your screen into an AI scanner.")
    with c2:
        st.warning("**2. Predict**")
        st.write("Click 'Predict'. The AI analyzes the image and returns the most likely animal with a confidence score.")
    with c3:
        st.success("**3. Feedback**")
        st.write("Provide feedback (👍/👎) to help us track performance. View your past scans in the History page.")

    st.markdown("---")

    # 5. Future Vision
    st.header("🔮 Future Vision")
    st.write("""
    We aim to continuously evolve this project. Our roadmap includes:
    - **Mobile App**: Launching a React Native version for on-the-go scanning.
    - **Real-time Video**: Recognizing animals in live video feeds.
    - **Wikipedia Integration**: Showing facts and habitat info for predicted animals.
    - **Community Leaderboard**: Gamifying the experience for top contributors.
    """)
    
    st.markdown("---")
    st.caption("© 2025 AI Animal Classifier Team. Built with ❤️ and Python.")

if __name__ == "__main__":
    run()
