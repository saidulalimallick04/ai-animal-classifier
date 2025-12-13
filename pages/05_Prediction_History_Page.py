# pages/05_Prediction_History_Page.py
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from core.sidebar import load_sidebar

PREDICTION_HISTORY_DIR = Path("prediction-history")

def run():
    st.title("🕒 Prediction History")
    st.write("View past predictions and their details.")
    st.markdown("---")

    # 1. Load model from Sidebar to filter history
    model_path = load_sidebar()
    if not model_path:
        st.info("👈 Please select a model from the sidebar to view its history.")
        return

    model_name = model_path.name
    st.subheader(f"History for: `{model_name}`")

    # 2. Check for History Directory and CSV
    model_history_dir = PREDICTION_HISTORY_DIR / model_name
    csv_path = model_history_dir / "prediction-history.csv"

    if not model_history_dir.exists() or not csv_path.exists():
        st.info(f"No history found for **{model_name}**. Try making some predictions first!")
        return

    # 3. Load and Display CSV
    try:
        # Use python engine and skip bad lines to handle header mismatches gracefully
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
        # Sort by Timestamp descending if possible
        if 'Timestamp' in df.columns:
            df = df.sort_values(by='Timestamp', ascending=False)
        
        # Create Tabs
        tab_log, tab_gallery = st.tabs(["History Log", "History Gallery"])
        
        with tab_log:
             with st.expander("History Log", expanded=True):
                st.dataframe(df, width="stretch", height=400)
        
        with tab_gallery:
            # 4. Image Gallery (Last 50)
            st.subheader("Recent Predictions Gallery (Last 50)")
            
            # Take top 50
            recent_df = df.head(50)
            
            if not recent_df.empty and 'Image Filename' in recent_df.columns:
                # Create 3 columns
                cols = st.columns(3)
                
                for i, (_, row) in enumerate(recent_df.iterrows()):
                    col = cols[i % 3]
                    
                    filename = row['Image Filename']
                    image_path = model_history_dir / filename
                    label = row['Predicted Label']
                    conf = float(row['Confidence']) if 'Confidence' in row else 0.0
                    timestamp = row['Timestamp']
                    
                    with col:
                        if image_path.exists():
                            img = Image.open(image_path)
                            st.image(img, width=300)
                            st.caption(f"**{label}** ({conf:.2%})\n\n*{timestamp}*")
                        else:
                            st.warning(f"Image missing: {filename}")
            else:
                st.info("No images to display.")
                     
    except Exception as e:
        st.error(f"Error loading history: {e}")

if __name__ == "__main__":
    run()
