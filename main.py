"""Streamlit Main Entry Point."""

from app.stt_model import STTModel
from app.ui import render_ui

def main() -> None:
    """Main function to run the Streamlit app."""
    stt_model = STTModel()
    render_ui(stt_model=stt_model)

if __name__ == "__main__":
    main()