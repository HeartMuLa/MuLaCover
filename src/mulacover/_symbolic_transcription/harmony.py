"""Audio-to-chord inference without the upstream training runtime."""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from .chord.decoder import ChordDecoder
from .chord.model import ChordNet


class ChordTranscriber:
    sample_rate = 22050
    hop_length = 512
    checkpoint_names = tuple(
        f"joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{fold}.best.sdict"
        for fold in range(5)
    )

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device(device)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("Chord dtype must be float32, float16, or bfloat16")
        if self.device.type == "cpu" and dtype != torch.float32:
            raise ValueError("Chord transcription on CPU requires float32")
        self.dtype = dtype
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_paths = [checkpoint_dir / name for name in self.checkpoint_names]
        for path in checkpoint_paths:
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing chord transcription checkpoint: {path}"
                )
        self.models = []
        for path in checkpoint_paths:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = ChordNet()
            model.load_state_dict(checkpoint["net"], strict=True)
            self.models.append(model.to(device=self.device, dtype=self.dtype).eval())
        self.decoder = ChordDecoder()

    @torch.inference_mode()
    def transcribe(
        self, audio_path: Union[str, Path], bpm: float
    ) -> List[Tuple[str, int, int]]:
        """Return complete chord labels and rounded whole-beat intervals."""
        if not np.isfinite(bpm) or bpm <= 0:
            raise ValueError("bpm must be positive and finite")
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        if audio.size == 0:
            raise ValueError("Reference audio is empty")
        cqt = librosa.hybrid_cqt(
            audio,
            sr=self.sample_rate,
            bins_per_octave=36,
            fmin=librosa.note_to_hz("F#0"),
            n_bins=288,
            tuning=None,
            hop_length=self.hop_length,
        ).T
        features = torch.from_numpy(np.abs(cqt).astype(np.float32)).to(
            device=self.device, dtype=self.dtype
        )
        predictions = [model.inference(features) for model in self.models]
        probabilities = [
            np.mean([prediction[head] for prediction in predictions], axis=0)
            for head in range(6)
        ]
        segments = self.decoder.decode_segments(
            probabilities, self.hop_length / self.sample_rate
        )
        seconds_per_beat = 60.0 / bpm
        return [
            (
                label,
                int(np.round(onset / seconds_per_beat)),
                int(np.round(offset / seconds_per_beat)),
            )
            for label, onset, offset in segments
        ]
