# Minutes of the meeting generator

A modular, well-documented, and ruff-compliant Streamlit application for transcribing audio to text using Hugging Face models.

## Features

- Drag-and-drop audio upload (wav, mp3, flac, ogg) via Streamlit UI
- Fast and accurate speech recognition using Hugging Face's `transformers` library (default: Whisper)
- Modular codebase with docstrings and type hints
- Includes unit tests and environment setup

## Setup

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

## Customization

- You can change the Hugging Face model in `STTModel` if you want to use a different model.
- Extend `tests/` for additional test coverage.

## Ruff Compliance

- All code is formatted and linted using [ruff](https://github.com/astral-sh/ruff).
