"""Test cases for STTModel."""

import io
import numpy as np
import soundfile as sf
from app.stt_model import STTModel


def test_transcribe_dummy(monkeypatch):
    """Test the transcribe method with a dummy audio input."""

    class DummyPipe:
        def __call__(self, audio, sampling_rate=None):
            return {"text": "dummy transcript"}

    stt = STTModel()
    stt._pipe = DummyPipe()

    # Create a dummy audio file in memory
    audio_array = np.zeros(16000)
    buf = io.BytesIO()
    sf.write(buf, audio_array, 16000, format="WAV")
    buf.seek(0)

    result = stt.transcribe(buf)
    assert result == "dummy transcript"