import streamlit as st
import os
from .diarization import DiarizationModel
from .conversation_builder import align_transcript_with_speakers, format_conversation
from .summary import ConversationSummarizer
def load_hf_token(token_file_path="conf/hf_token.txt"):
    """Load Hugging Face token from Streamlit secrets or local file."""
    # Prefer Streamlit Secrets in cloud deployments
    try:
        token = st.secrets.get("HF_TOKEN")  # type: ignore[attr-defined]
        if token:
            return str(token)
    except Exception:
        pass

    # Fallback to local file for dev/local runs
    token_file_path = os.path.join(os.path.dirname(__file__), "..", token_file_path)
    try:
        with open(token_file_path, "r") as f:
            return f.read().strip()
    except Exception:
        print("Unable to read the token.")
        return None

def render_ui(stt_model):
    st.title("Speech-to-Text with Speaker Diarization")
    st.caption("Transcribe audio, attribute speakers, and summarize the conversation.")

    # Sidebar controls
    st.sidebar.header("Settings")
    num_speakers = st.sidebar.number_input("Number of Speakers", min_value=1, max_value=10, value=2, step=1)
    enable_diarization = st.sidebar.checkbox("Enable Diarization", value=True)

    # File uploader and preview
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")

    # Load Hugging Face token
    hf_token = load_hf_token()
    if enable_diarization and not hf_token:
        st.warning("Hugging Face access token not found. Diarization requires a valid token.")

    # Process button
    if st.button("Process", type="primary", disabled=audio_file is None):
        if audio_file is None:
            st.info("Please upload an audio file to get started.")
            return

        # Read the file bytes once
        audio_bytes = audio_file.read()

        # Steps columns for better layout
        status_col, output_col = st.columns([1, 2])

        with status_col:
            st.subheader("Status")

            # Diarization (optional)
            diarization_segments = []
            if enable_diarization:
                if not hf_token:
                    st.error("Missing HF token. Skipping diarization.")
                else:
                    with st.spinner("Diarizing speakers (pyannote)..."):
                        try:
                            diarizer = DiarizationModel(access_token=hf_token, num_speakers=num_speakers)
                            diarization_segments = diarizer.diarize(audio_bytes)
                            st.success("Diarization complete")
                        except Exception as e:
                            st.error(f"Diarization failed: {e}")
                            diarization_segments = []

            # Transcription
            with st.spinner("Transcribing audio (Whisper)..."):
                try:
                    transcript_chunks = stt_model.transcribe(audio_bytes)
                    st.success("Transcription complete")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return

            # Alignment
            with st.spinner("Aligning transcript with speakers..."):
                try:
                    if enable_diarization and diarization_segments:
                        conversation = align_transcript_with_speakers(transcript_chunks, diarization_segments)
                    else:
                        # No diarization: attribute all to a single speaker
                        conversation = [
                            {
                                "start": ch.get("timestamp", [0, 0])[0] if ch.get("timestamp") else ch.get("start", 0),
                                "end": ch.get("timestamp", [0, 0])[1] if ch.get("timestamp") else ch.get("end", 0),
                                "speaker": "Speaker",
                                "text": ch.get("text", ""),
                            }
                            for ch in transcript_chunks
                        ]
                    conversation_text = format_conversation(conversation)
                    st.success("Alignment complete")
                except Exception as e:
                    st.error(f"Alignment failed: {e}")
                    return

            # Summarization
            with st.spinner("Generating summary..."):
                try:
                    summarizer = ConversationSummarizer()
                    summary = summarizer.summarize(conversation_text)
                    st.success("Summary generated")
                except Exception as e:
                    st.error(f"Summarization failed: {e}")
                    summary = ""

        with output_col:
            st.subheader("Summary")
            st.write(summary if summary else "No summary available.")

            st.subheader("Conversation")
            st.text(conversation_text)

            # Download buttons
            if summary:
                st.download_button("Download Summary", data=summary, file_name="summary.txt", mime="text/plain")
            st.download_button("Download Conversation", data=conversation_text, file_name="conversation.txt", mime="text/plain")
    else:
        st.info("Upload audio and click Process to begin.")
