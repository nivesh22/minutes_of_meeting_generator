# app Module

This folder contains the core logic for the Speech-to-Text application.

- `stt_model.py`: Implements the `STTModel` class which loads a Hugging Face speech recognition model and provides a `transcribe` method to convert audio files to text.
- `ui.py`: Contains the Streamlit UI logic, allowing users to drag and drop audio files for transcription.
- `Diarization.py`: Implements speaker diarization using a pre-trained model (such as pyannote.audio).
- It segments the audio by identifying and labeling different speakers, allowing each portion of the transcript to be associated with the correct speaker.
- `Summary.py`: Generates concise meeting summaries from the full transcript using a language model.
- It extracts key discussion points, action items, and decisions, presenting them in a structured minutes-of-meeting format.

## Workflow

**Model Initialization:**
STTModel is initialized with a pre-trained Hugging Face model (default: openai/whisper-base).

**Audio Upload:**
The Streamlit UI (ui.py) allows users to upload an audio file via drag-and-drop.

**Speaker Diarization:**
The uploaded audio is processed by the Diarization module to detect and label speakers, segmenting the audio into chunks per speaker.

**Transcription:**
Each speaker segment is transcribed using the transcribe method from STTModel, producing a speaker-attributed transcript.

**Summarization:**
The Summary module takes the combined transcript and generates a structured summary including key points, decisions, and action items.

**Display and Export:**
The UI displays both the full transcript (with speaker labels) and the generated summary, allowing users to download the final minutes of the meeting in text or document format.
