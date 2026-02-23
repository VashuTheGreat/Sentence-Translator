# 🌐 Sentence-Translator: English to Hindi

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)

A robust, enterprise-grade English-to-Hindi translation application built with a Sequence-to-Sequence (Seq2Seq) architecture and modern MLOps principles.

## 🚀 Key Features

- **Advanced Deep Learning**: Implements a GRU-based Encoder-Decoder architecture with teacher forcing.
- **Memory-Mapped Data (Scale-Ready)**: Custom `np.memmap` integration for data transformation and training, allowing the pipeline to handle datasets far larger than available RAM.
- **Modular MLOps Pipeline**:
  - **Data Ingestion**: Automated fetching and ingestion.
  - **Data Validation**: Schema and quality checks.
  - **Data Transformation**: Fixed-width tokenization and memory-mapped storage.
  - **Model Training**: Scalable training with configurable hyperparameters.
  - **Prediction**: Robust inference engine with architecture reconstruction.
- **Live Tracking**: Integrated with **MLflow** and **DagsHub** for experiment tracking.
- **Modern UI**: Interactive **Streamlit** dashboard for real-time translation.

## 📊 Live Experiment Tracking

Monitor training metrics and model performance here:
[DagsHub MLflow Tracking](https://dagshub.com/vanshsharma7832/Sentence-Translator.mlflow/#/)

## 🛠️ Tech Stack

- **Core**: PyTorch, NumPy, Pandas
- **Experiment Tracking**: MLflow, DagsHub
- **Platform**: Streamlit
- **Dependency Management**: `uv`
- **Data Versioning**: DVC

## 📁 Project Structure

```text
├── config/                 # YAML configs for training and validation
├── src/
│   ├── components/        # Pipeline components (Ingestion, Training, etc.)
│   ├── entity/           # Data classes for artifacts and configs
│   ├── pipelines/        # Process-specific workflow pipelines
│   └── utils/            # Shared utilities (Main utils, Async handler)
├── saved_model/            # Production-ready model and vocab artifacts
├── StreamlitApp/           # Interactive web application
└── notebooks/              # Research and experimentation
```

## 🏗️ Getting Started

### 1. Installation

Using `uv` for lightning-fast setup:

```powershell
uv sync
```

### 2. Training the Pipeline

To run the full end-to-end training process:

```powershell
uv run python main.py
```

### 3. Running Real-time Predictions

Verify the prediction engine with a sample script:

```powershell
uv run python src/tests/test_prediction_fix.py
```

### 4. Launch the Web App

```powershell
uv run streamlit run StreamlitApp/app.py
```

## 🧠 Technical Highlights: Memory Map Optimization

To prevent "Out of Memory" errors during large-scale training, this project uses **Memory-Mapped Files (.dat)**. Instead of loading the entire tokenized dataset into RAM, we map the files directly to disk using `np.memmap`, ensuring nearly constant memory usage regardless of dataset size.

---

Developed with ❤️ by Vansh Sharma
