from __future__ import annotations

"""Conversation summarization utilities.

This module provides a ConversationSummarizer that summarizes long
conversation transcripts using a local Transformers summarization model.

Design goals:
- Prefer local model assets (offline-friendly) if available.
- Handle long inputs by chunking based on tokenizer limits.
- Provide a safe, lightweight fallback when the model is unavailable.
"""

from typing import List, Optional

import os


class ConversationSummarizer:
    """Summarize conversation text into a concise summary.

    By default, loads a local BART CNN model from `bart-large-cnn` at the
    project root to avoid network access. If loading fails, falls back to a
    simple extractive heuristic.
    """

    def __init__(
        self,
        model_dir: str = "bart-large-cnn",
        device: Optional[str] = None,
        min_length: int = 30,
        max_length: int = 180,
    ) -> None:
        """Initialize the summarizer.

        Args:
            model_dir: Directory containing the local summarization model
                (e.g., a local copy of `facebook/bart-large-cnn`). Resolved
                relative to the project root.
            device: Optional device hint: "cuda" or "cpu". If None, auto-detects.
            min_length: Minimum summary length per chunk (tokens, model-dependent).
            max_length: Maximum summary length per chunk (tokens, model-dependent).
        """
        self._pipe = None
        self.min_length = min_length
        self.max_length = max_length

        # Resolve model path relative to repo root.
        # app/summary.py -> go up one directory, then join model_dir
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        resolved_model_path = os.path.join(root_dir, model_dir)

        # Lazy import transformers to keep import surface small in environments
        # where it might be unavailable at import time.
        try:
            from transformers import pipeline  # type: ignore
            import torch  # type: ignore

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use local model assets if available; otherwise allow HF hub resolution
            # (though network may be restricted in this environment).
            model_path = resolved_model_path if os.path.isdir(resolved_model_path) else model_dir

            self._pipe = pipeline(
                "summarization",
                model=model_path,
                tokenizer=model_path,
                device=0 if device == "cuda" else -1,
            )
        except Exception:
            # If anything fails (no transformers, missing model, etc.)
            # we keep _pipe as None and rely on the fallback path.
            self._pipe = None

    def summarize(self, text: str) -> str:
        """Summarize a conversation text.

        Splits long inputs into tokenizer-aware chunks and summarizes each
        chunk, then optionally summarizes the concatenated chunk summaries.

        Args:
            text: The full conversation text, e.g., lines of "Speaker: utterance".

        Returns:
            A concise summary string.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return "No content to summarize."

        if self._pipe is None:
            return self._fallback_summary(cleaned)

        # Token-aware chunking based on the pipeline's tokenizer limits.
        tokenizer = getattr(self._pipe, "tokenizer", None)
        model_max = getattr(tokenizer, "model_max_length", 1024) or 1024

        chunks = self._chunk_by_tokens(cleaned, max_tokens=max(256, min(900, model_max - 64)))
        partial_summaries: List[str] = []

        for chunk in chunks:
            try:
                out = self._pipe(
                    chunk,
                    do_sample=False,
                    min_length=self.min_length,
                    max_length=self.max_length,
                )
                if isinstance(out, list) and out and "summary_text" in out[0]:
                    partial_summaries.append(out[0]["summary_text"].strip())
                else:
                    partial_summaries.append(self._safe_truncate(chunk))
            except Exception:
                partial_summaries.append(self._safe_truncate(chunk))

        if not partial_summaries:
            return self._fallback_summary(cleaned)

        # If we had multiple chunks, briefly summarize their concatenation.
        combined = "\n".join(partial_summaries)
        if len(partial_summaries) == 1:
            return combined

        try:
            out = self._pipe(
                combined,
                do_sample=False,
                min_length=max(20, self.min_length // 2),
                max_length=max(60, self.max_length // 2),
            )
            if isinstance(out, list) and out and "summary_text" in out[0]:
                return out[0]["summary_text"].strip()
        except Exception:
            pass

        return combined

    def _chunk_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within `max_tokens` using the model tokenizer.

        Falls back to a simple line-based chunking if tokenizer is unavailable.
        """
        tokenizer = getattr(self._pipe, "tokenizer", None)
        if tokenizer is None:
            # Simple fallback: chunk by lines with a character cap as a proxy
            return self._chunk_by_lines(text, max_chars=max_tokens * 4)

        lines = [ln for ln in text.splitlines() if ln.strip()]
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for ln in lines:
            # Estimate tokens for the next line.
            try:
                tok_len = len(tokenizer.encode(ln, add_special_tokens=False))
            except Exception:
                tok_len = max(1, len(ln) // 4)

            # If adding this line would overflow, flush current chunk.
            if current and current_len + tok_len > max_tokens:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            current.append(ln)
            current_len += tok_len

        if current:
            chunks.append("\n".join(current))

        return chunks if chunks else [text]

    @staticmethod
    def _chunk_by_lines(text: str, max_chars: int) -> List[str]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        chunks: List[str] = []
        buf: List[str] = []
        length = 0
        for ln in lines:
            if buf and length + len(ln) + 1 > max_chars:
                chunks.append("\n".join(buf))
                buf = []
                length = 0
            buf.append(ln)
            length += len(ln) + 1
        if buf:
            chunks.append("\n".join(buf))
        return chunks if chunks else [text]

    @staticmethod
    def _safe_truncate(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _fallback_summary(self, text: str) -> str:
        """Provide a simple extractive fallback summary without ML models."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return "No content to summarize."
        # Take first few salient lines as a crude summary.
        head = lines[:5]
        return "\n".join(head)
