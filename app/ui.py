import streamlit as st
import os
from .diarization import DiarizationModel
from .conversation_builder import align_transcript_with_speakers, format_conversation
from .summary import ConversationSummarizer
def load_hf_token(token_file_path="conf/hf_token.txt"):
    token_file_path = os.path.join(os.path.dirname(__file__), "..", token_file_path)
    try:
        with open(token_file_path, "r") as f:
            return f.read().strip()
    except Exception:
        print("Unable to read the token.")
        return None

def render_ui(stt_model):
    st.title("Speech-to-Text with Speaker Diarization")

    # Sidebar for number of speakers
    num_speakers = st.sidebar.number_input("Number of Speakers", min_value=1, max_value=10, value=2, step=1)

    # File uploader
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    # Automatically load Hugging Face token from file
    hf_token = load_hf_token()
    if not hf_token:
        st.warning("Hugging Face access token not found or invalid.")
        return

    # Submit button
    if audio_file is not None and hf_token:
        if st.button("Submit"):
            # Read the file bytes once
            audio_bytes = audio_file.read()

            # -- Diarization --
            st.info("Diarizing speakers (pyannote)...")
            diarizer = DiarizationModel(access_token=hf_token, num_speakers=num_speakers)
            diarization_segments = diarizer.diarize(audio_bytes)

            # -- Transcription --
            st.info("Transcribing audio (Whisper)...")
            transcript_chunks = stt_model.transcribe(audio_bytes)



            # -- Conversation alignment --
            st.info("Aligning transcript with speakers...")
            conversation = align_transcript_with_speakers(transcript_chunks, diarization_segments)
            conversation_text = format_conversation(conversation)
            
            st.info("Generating Summary...")
            summarizer = ConversationSummarizer()
            summary = summarizer.summarize(conversation_text)

            st.subheader("Summary")
            st.text(summary)

            st.subheader("Conversation")
            st.text(conversation_text)
    elif audio_file is not None and not hf_token:
        st.warning("Hugging Face access token not found in hf_token.txt.")
    else:
        st.write("Please upload an audio file to get started.")
