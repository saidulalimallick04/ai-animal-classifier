# 📅 Project Timeline & Changelog

This document tracks the development history of the **AI-Animal-Classifier** project, detailing weekly progress, major milestones, and architectural decisions from inception (September 2025) to the present.

---

## 🟢 December 2025 (Refinement & Optimization)

### Week 3: Metadata modernization & Cloud Prep (Current)

- **JSON Migration**: Fully migrated model metadata storage from legacy CSV to JSON (`model-details.json`) to fix parsing issues and support structured data.
- **Live Prediction Overhaul**: Refactored `pages/02_Live_Prediction_Page.py` to use a unified prediction pipeline. Added a 2-column layout for the camera and image previews for uploads.
- **Deep Learning Guide**: Created comprehensive documentation (`DEEP_LEARNING_GUIDE.md`) explaining CNNs, MobileNetV2, and Transfer Learning.
- **Google Drive Integration**: Added `g_drive_file_id` schema to support future auto-downloading of models.
- **Blueprint Update**: Standardized project documentation and verified `pyproject.toml` dependencies.

### Week 2: Core Refactoring

- **Modularization**: Extracted logic from `main.py` into a dedicated `core/` package (`model.py`, `sidebar.py`, `prediction.py`, `history.py`).
- **Sidebar Upgrades**: Implemented a dynamic model selector that scans the file system and reads metadata.
- **Warning Suppression**: Addressed TensorFlow/oneDNN verbose logging to clean up the console output.
- **History Tracking**: Implemented `pages/05_Prediction_History_Page.py` to save and display past predictions with user feedback.

### Week 1: Interface Polish

- **Streamlit Multipage Mode**: Restructured the app from a single script into a multi-page app using `st.navigation`.
- **Dashboard**: Created `pages/01_Dashboard_Page.py` for a unified "Upload & Predict" experience.
- **Theming**: Applied custom page configs (`layout="wide"`, custom titles) for a more professional look.

---

## 🟠 November 2025 (Web Interface Development)

### Week 4: First Streamlit Prototype

- **Initial App**: Built `main.py` allowing users to upload an image and get a categorical prediction.
- **UI Basics**: Added "Home" and "About" sections.
- **Integration**: Connected the trained `.h5` models to the web frontend using `tensorflow.keras.models.load_model`.

### Week 3: Model Inference Scripts

- **CLI Tools**: Created standalone Python scripts to test model inference on local images without a UI.
- **Preprocessing Pipeline**: Standardized image resizing (224x224) and normalization logic.
- **Error Handling**: Added checks for corrupt images and invalid file formats.

### Week 2: Model Zoo Expansion

- **Binary Model**: Trained a lightweight `SimpleCNN` for specific binary classification tasks (e.g., Cat vs Dog) to compare performance.
- **Model Registry**: Started tracking model metrics (Accuracy, Loss) in a simple CSV file (later replaced by JSON).

### Week 1: Streamlit Research

- **Framework Selection**: Evaluated Flask vs Streamlit. Chose Streamlit for rapid prototyping and ease of Python integration.
- **Environment Setup**: Configured `requirements.txt` with `streamlit`, `altair`, and `pillow`.

---

## 🟡 October 2025 (Advanced Model Training)

### Week 4: Transfer Learning Breakthrough

- **MobileNetV2**: Adopted Transfer Learning using the MobileNetV2 architecture pre-trained on ImageNet.
- **Performance**: Achieved >90% accuracy on the test set, significantly outperforming custom CNNs.
- **Optimization**: Tuned hyperparameters (Learning Rate: 1e-4, Adam Optimizer).

### Week 3: Custom CNN Experiments

- **Architecture Design**: Experimented with custom CNN layers (Conv2D -> MaxPooling -> Flatten -> Dense).
- **Overfitting**: Encountered high variance; added Dropout layers and Data Augmentation (Rotation, Zoom) to mitigate.

### Week 2: Dataset Expansion

- **Data Collection**: Scraped and curated a larger dataset covering 99 animal classes.
- **Cleaning**: Removed duplicates and non-animal images using hash-based filtering. (Tools: `hashlib`, `PIL`).

### Week 1: Training Pipeline Setup

- **Categorical Script**: Developed `scripts/categorical-model-gen-script.py` to handle data loading and model training loops.
- **Callbacks**: Implemented `ModelCheckpoint` and `EarlyStopping` to save the best model and prevent wasted compute.

---

## 🟣 September 2025 (Inception & Fundamentals)

### Week 4: Binary Classification

- **First Model**: Successfully trained a basic model to distinguish between 2 classes.
- **Metrics**: Learned to interpret Confusion Matrices and Accuracy scores.

### Week 3: Deep Learning Basics

- **Theoretical Study**: Research on Neural Networks, Backpropagation, and Convolutions.
- **TensorFlow Setup**: Installed TensorFlow/Keras and configured GPU support (CUDA/cuDNN).

### Week 2: Data Preprocessing Logic

- **Image Operations**: Wrote utility functions to resize images and convert them to NumPy arrays.
- **Label Encoding**: Implemented One-Hot Encoding for categorical labels.

### Week 1: Project Kickoff

- **Concept**: Defined the goal of building an "AI Animal Classifier" usable by non-experts.
- **Scope**: Targeted a web-based deployment with a focus on ease of use.
- **Repo Init**: Initialized Git repository and set up the basic folder structure (`scripts/`, `data/`).

---

## 👨‍💻 Author

| Profile | Developer Name | Role | GitHub | LinkedIn | X |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [![Sami](https://github.com/saidulalimallick04.png?size=75)](https://github.com/saidulalimallick04) | Saidul Ali Mallick (Sami) | Backend Developer & AIML Engineer & Team Lead | [@saidulalimallick04](https://github.com/saidulalimallick04) | [@saidulalimallick04](https://linkedin.com/in/saidulalimallick04) | [@saidulmallick04](https://x.com/saidulmallick04) |

> ❤️ I believe in building impact, not just writing code.
> _💚 Backend Sage signing off.._

---
