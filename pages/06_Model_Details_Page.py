# pages/06_Model_Details_Page.py
import streamlit as st
import pandas as pd
from core.model import get_model_metadata, select_model

def run():
    st.title("📝 Model Metadata Registry")
    st.write("Overview of all available models and their configurations.")
    st.markdown("---")

    # 1. Fetch Metadata
    df = get_model_metadata()

    if df.empty:
        st.warning("No model metadata found in Registry (JSON).")
        return

    # Create Main Tabs
    main_tab1, main_tab2 = st.tabs(["📊 Model Registry", "🔍 Deep Dive Analysis"])
    
    # --- Tab 1: Registry ---
    with main_tab1:
        st.subheader("All Models")
        st.dataframe(df, width="stretch", height=400)
    
    # --- Tab 2: Deep Dive ---
    with main_tab2:
        st.subheader("Deep Dive")
        model_names = df['model_name'].tolist() if 'model_name' in df.columns else []
        
        selected_model_name = st.selectbox("Select a specific model to inspect parameters:", ["-- Select --"] + model_names)
        
        if selected_model_name and selected_model_name != "-- Select --":
            row = df[df['model_name'] == selected_model_name].iloc[0]
            
            # Tabs for better organization within Deep Dive
            tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "⚙️ Architecture & Params", "📈 Performance", "🏷️ Output Labels"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Model Name:** `{row.get('model_name', 'N/A')}`")
                    st.markdown(f"**Model Type:** `{row.get('model_type', 'N/A')}`")
                    st.markdown(f"**Prediction Type:** `{row.get('prediction_type', 'N/A')}`")
                with c2:
                    st.markdown(f"**Output Categories:** `{row.get('no_of_output_categories', 'N/A')}`")
                    st.markdown(f"**Input Shape:** `{row.get('input_shape', 'N/A')}`")

            with tab2:
                # ... existing tab2 code ...
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Optimizer", row.get('optimizer', 'N/A'))
                with c2:
                    st.metric("Learning Rate", row.get('learning_rate', 'N/A'))
                with c3:
                    st.metric("Epochs", row.get('epochs', 'N/A'))
                
                st.markdown("### Architecture Description")
                st.write(row.get('architecture', 'No detailed description available.'))
                
            with tab3:
               # ... existing tab3 code ...
                c1, c2 = st.columns(2)
                with c1:
                    train_acc = row.get('train_accuracy', 0)
                    st.metric("Train Accuracy", f"{train_acc}")
                with c2:
                    val_acc = row.get('validation_accuracy', 0)
                    st.metric("Validation Accuracy", f"{val_acc}")
            
            with tab4:
                st.subheader("Output Labels")
                labels_str = row.get('output_labels', '')
                
                if pd.notna(labels_str) and labels_str:
                    # Clean and split
                    clean_labels = labels_str.replace('"', '').replace("'", "")
                    label_list = [x.strip() for x in clean_labels.split(',') if x.strip()]
                    
                    # Display in a grid (e.g., 4 columns)
                    cols_count = 4
                    cols = st.columns(cols_count)
                    
                    for i, label in enumerate(label_list):
                        col = cols[i % cols_count]
                        with col:
                            st.markdown(f"- {label.title()}")
                else:
                     st.info("No labels information available for this model.")

if __name__ == "__main__":
    run()