# Speech-to-Text App

A modular, well-documented, and ruff-compliant Streamlit application for transcribing audio to text using Hugging Face models.

## Features

- Drag-and-drop audio upload (wav, mp3, flac, ogg) via Streamlit UI
- Fast and accurate speech recognition using Hugging Face's `transformers` library (default: Whisper)
- Modular codebase with docstrings and type hints
- Includes unit tests and environment setup

## Setup (Local)

1. **Clone the repository**
2. **Create environment**:
    ```bash
    conda env create -f environment.yml
    conda activate speech-to-text-env
    ```

3. **Run the app**:
    ```bash
    streamlit run main.py
    ```

## Structure

- `app/`: Core modules for model and UI
- `tests/`: Unit tests
- `main.py`: Entry point for Streamlit
- `environment.yml`: Dependencies
- `README.md`: Overview and instructions
 - `pages/`: Multipage Streamlit pages (e.g., About)
 - `.streamlit/config.toml`: App theming

## Deployment (Streamlit Cloud)

- Add `requirements.txt` or use the provided one.
- In Streamlit Cloud, set Secrets (left sidebar → App settings → Secrets):
  - `HF_TOKEN`: Your Hugging Face access token (required for diarization).
- Optional environment variables:
  - `ASR_MODEL_NAME`: Whisper model name (e.g., `openai/whisper-small`).
  - `ASR_MODEL_LOCAL_DIR`: Path to a vendored model directory in your repo.

### Models in the cloud

- ASR (Whisper): The app reads `ASR_MODEL_LOCAL_DIR` first. If set and exists, it loads the model from that directory. Otherwise it uses `ASR_MODEL_NAME` (defaults to `openai/whisper-medium`).
- Summarizer (BART CNN): Loads from the local `bart-large-cnn/` directory if present; otherwise falls back to a lightweight extractive summary.

## Customization

- You can change the Hugging Face model in `STTModel` if you want to use a different model.
- Extend `tests/` for additional test coverage.

## Ruff Compliance

- All code is formatted and linted using [ruff](https://github.com/astral-sh/ruff).
