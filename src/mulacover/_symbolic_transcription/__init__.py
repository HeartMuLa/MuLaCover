"""Private, lazily loaded audio-to-symbolic inference backends."""

import gc
import math
from pathlib import Path

import torch

from ..symbolic import SymbolicCondition


class SymbolicTranscriber:
    def __init__(self, checkpoint_dir, device, dtype=torch.float32, lazy_load=True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.lazy_load = lazy_load
        self._melody = None
        self._harmony = None

    def _release(self, name):
        if self.lazy_load:
            setattr(self, name, None)
            gc.collect()
            if self.device.type == "cuda" and torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    @torch.inference_mode()
    def transcribe(self, audio_path, bpm=None):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError(
                "Audio transcription requires pip install 'mulacover[audio]'"
            ) from exc

        if bpm is None:
            # The reference frontend estimates tempo at the source sample rate, not librosa's
            # default 22050 Hz. Both note and chord quantization use this BPM.
            audio, sample_rate = librosa.load(audio_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            bpm = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)
        if not isinstance(bpm, (float, int)) or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(
                "Could not determine a positive BPM; provide bpm explicitly"
            )
        melody_checkpoint = self.checkpoint_dir / "yourmt3" / "last.ckpt"
        harmony_dir = self.checkpoint_dir / "chord"
        if not melody_checkpoint.is_file():
            raise FileNotFoundError(
                f"YourMT3 checkpoint not found: {melody_checkpoint}"
            )
        if not harmony_dir.is_dir():
            raise FileNotFoundError(
                f"Chord checkpoint directory not found: {harmony_dir}"
            )

        from .melody import MelodyTranscriber
        from .harmony import ChordTranscriber

        try:
            if self._melody is None:
                self._melody = MelodyTranscriber(
                    melody_checkpoint, self.device, self.dtype
                )
            notes = self._melody.transcribe(audio_path)
        finally:
            self._release("_melody")
        try:
            if self._harmony is None:
                self._harmony = ChordTranscriber(harmony_dir, self.device, self.dtype)
            chords = self._harmony.transcribe(audio_path, bpm)
        finally:
            self._release("_harmony")
        return SymbolicCondition.from_transcription(notes, chords, bpm)
