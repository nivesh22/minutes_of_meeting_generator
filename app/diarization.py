import torch
import torchaudio
import tempfile
from pyannote.audio import Pipeline

class DiarizationModel:
    def __init__(self, access_token, num_speakers=1):
        try:
            # Set device globally
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.set_default_device(device)

            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=access_token
            )
            self.num_speakers = num_speakers
        except Exception as e:
            print(f"Error loading pyannote pipeline: {e}")
            self.pipeline = None

    def diarize(self, audio_bytes):
        if self.pipeline is None:
            raise ValueError("Diarization pipeline is not initialized.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name

        waveform, rate = torchaudio.load(temp_file_path)
        if rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)
            waveform = resampler(waveform)
        mono_waveform = waveform.mean(dim=0, keepdim=True)  # shape: (1, time)


        def progress_callback(progress):
            print(f"Diarization progress: {progress * 100:.2f}%")

        diarization = self.pipeline(
            {"waveform": mono_waveform, "sample_rate": 16000},
            num_speakers=self.num_speakers,
            # progress_hook=progress_callback
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return segments
