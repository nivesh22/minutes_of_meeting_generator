import torchaudio
from transformers import pipeline
import torch
import tempfile

class STTModel:
    def __init__(self, model_name="openai/whisper-medium", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def transcribe(self, audio_bytes) -> list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        waveform, rate = torchaudio.load(temp_file_path)

        if rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert to mono if needed
        audio = waveform.mean(dim=0).numpy()

        # Run transcription with timestamp output
        result = self.pipe(audio, return_timestamps="chunks", generate_kwargs={"language": "en", "task": "transcribe"})

        if "chunks" not in result:
            raise ValueError("Transcription result does not contain 'chunks'.")
        return result["chunks"]
