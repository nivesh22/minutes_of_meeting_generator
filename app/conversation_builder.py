def align_transcript_with_speakers(transcript_chunks, diarization_segments):
    aligned = []

    for chunk in transcript_chunks:
        timestamp = chunk.get("timestamp")

        # Fallback handling if "timestamp" key is missing or None
        if timestamp is not None:
            chunk_start, chunk_end = timestamp
        else:
            # Use safe .get() with defaults instead of direct indexing
            chunk_start = chunk.get("start", 0)
            chunk_end = chunk.get("end", 0)

        # Find speaker segment matching this time range
        speaker_label = "unknown"
        for segment in diarization_segments:
            if segment["start"] <= chunk_start <= segment["end"]:
                speaker_label = segment["speaker"]
                break

        aligned.append({
            "start": chunk_start,
            "end": chunk_end,
            "speaker": speaker_label,
            "text": chunk.get("text", "")
        })

    return aligned


def format_conversation(conversation):
    """
    Returns a human-readable conversation string.
    """
    lines = []
    for turn in conversation:
        lines.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(lines)
