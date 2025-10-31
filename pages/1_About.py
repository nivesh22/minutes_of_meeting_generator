import streamlit as st

# Page config should be the first Streamlit command on a page
st.set_page_config(
    page_title="About • Speech-to-Text + Diarization",
    page_icon="🗣️",
    layout="wide",
)

st.title("About This App")
st.markdown(
    """
This app converts speech to text, optionally performs speaker diarization to
attribute who said what, and then generates a concise summary of the
conversation.

Features:
- Upload audio files (`.wav`, `.mp3`, `.m4a`).
- Transcribe using Whisper via Hugging Face `transformers`.
- Optional speaker diarization powered by `pyannote.audio` (requires a Hugging Face token).
- Automatic conversation alignment and clean formatting.
- Summarization using a local BART CNN model for offline-friendly usage.

How to use:
1. Go to the main page and upload your audio file.
2. Toggle diarization on if you have a valid Hugging Face token configured.
3. Click **Process** to transcribe, align, and summarize.
4. Download the summary and conversation text.

Deployment notes:
- On Streamlit Cloud, set `HF_TOKEN` in Secrets to enable diarization.
- For ASR model control, set `ASR_MODEL_NAME` or point to a local folder via
  `ASR_MODEL_LOCAL_DIR` if you vendor the model in your repo.
- The summarizer loads a local `bart-large-cnn/` folder if present; otherwise it
  falls back to a simple extractive summary.
"""
)

st.info(
    "Tip: For faster cold starts in the cloud, prefer smaller ASR models like "
    "`openai/whisper-small` or vendor the model assets and use `ASR_MODEL_LOCAL_DIR`."
)

