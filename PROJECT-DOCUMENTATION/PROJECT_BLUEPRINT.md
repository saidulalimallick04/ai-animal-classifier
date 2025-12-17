# Project Blueprint for AI-Animal-Classifier

## Overview

The **AI-Animal-Classifier** project provides a Streamlit‑based web interface for classifying animal images using a deep learning model. The model is stored externally (Google Drive) due to its large size and is loaded at runtime.

## Technology Stack

- **Language**: Python 3.x
- **Web Framework**: Streamlit
- **Machine Learning**: TensorFlow, Keras, NumPy, Pandas, Scipy
- **Deployment**: Local run via `streamlit run main.py` or Docker.
- **Data Handling**: Pandas, JSON (for metadata)

## Key Libraries (from `pyproject.toml`)

| Library | Version Requirement |
|---|---|
| streamlit | >=1.52.1 |
| tensorflow | >=2.20.0 |
| pandas | >=2.3.3 |
| scipy | >=1.15.3 |
| gdown | >=5.2.0 |
| streamlit-local-storage | >=0.0.25 |

## Page Definitions

The application structure is defined in `main.py` using `st.navigation`:

| Page Title | Path | Icon | Description |
|---|---|---|---|
| **Home** | `/` | 🏠 | Landing page with overview and instructions. |
| **Dashboard** | `dashboard/` | 📊 | Unified prediction interface with feedback. |
| **Live Prediction** | `live-prediction/` | 🎥 | Camera and upload support for real-time inference. |
| **Prediction History** | `prediction-history/` | 📜 | Log of past predictions with user feedback. |
| **Model Details** | `model-details/` | 🛠️ | Technical specs of available models (from JSON). |
| **About** | `about/` | ℹ️ | Project information and credits. |
| **Old Interface** | `old-interface/` | 🔙 | Legacy prediction page (archived). |

## Recent Updates

**2025-12-14** (Newest First)

1. **Refined Live Prediction UI**: Optimized layout with 2-column camera view and added image preview for uploads.
2. **Google Drive Metadata**: Added `g_drive_file_id` to model schemas to support future cloud download capabilities.
3. **Live Prediction Refactor**: Updated `Live Prediction` page to use the unified `run_prediction_pipeline`, enabling persistent results and consistent feedback logic.
4. **JSON Metadata Migration**: Fully migrated model registry from CSV to `model-details.json` for better reliability.
5. **Cleaned Session Logic**: Reverted experimental session storage from the sidebar to ensure stability.
6. **Warning Suppression**: Globally suppressed noisy TensorFlow/oneDNN logs in `core/model.py`.

## Data Flow

1. User enters **Dashboard** or **Live Prediction**.
2. **Sidebar** loads model list from `prediction-model/model-details.json`.
3. User selects a model; app loads it (caching supported).
4. User provides image (Upload/Camera).
5. `core.prediction.run_prediction_pipeline` handles preprocessing, inference, and history logging.
6. Results and Confidence displayed; User can provide 👍/👎 feedback.

---

## 👨‍💻 Author

|Profile                                                                                                   | Member Name                   | Role                                              | GitHub                                                            | LinkedIn                                                          |
|----------------------------------------------------------------------------------------------------------|-------------------------------|---------------------------------------------------|-------------------------------------------------------------------|-------------------------------------------------------------------|
| [![Sami](https://github.com/saidulalimallick04.png?size=75)](https://github.com/saidulalimallick04)      | Saidul Ali Mallick (Sami)     | Backend Developer & AIML Engineer & Team Lead     | [@saidulalimallick04](https://github.com/saidulalimallick04)      | [@saidulalimallick04](https://linkedin.com/in/saidulalimallick04) |

> ❤️ I believe in building impact, not just writing code.
> _💚 Backend Sage signing off.._
