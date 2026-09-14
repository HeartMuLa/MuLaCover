"""Lazy YourMT3 melody and percussion transcription with explicit model resources."""

from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from .yourmt3.utils.note_event_dataclasses import Note


class MelodyTranscriber:
    """Transcribe notes in seconds, retaining singing voice and drum programs.

    The pretrained YourMT3 ``last.ckpt`` is loaded only when this object is
    constructed. Program 100 denotes lead singing; ``is_drum`` identifies
    percussion. Selecting conditioning tracks belongs to the caller.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device],
        dtype: torch.dtype = torch.float32,
    ):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"YourMT3 checkpoint not found: {checkpoint_path}")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("YourMT3 dtype must be float32, float16, or bfloat16")
        self.device = torch.device(device)
        self.dtype = dtype
        if self.device.type == "cpu" and dtype != torch.float32:
            raise ValueError("YourMT3 CPU transcription requires float32")

        from .yourmt3.model.inference import YourMT3
        from .yourmt3.checkpoint import load_state_dict

        state_dict = load_state_dict(checkpoint_path)
        self.model = YourMT3()
        self.model.load_state_dict(state_dict, strict=True)
        # The STFT remains float32, with optional mixed precision in neural layers.
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def transcribe(self, audio_path: Union[str, Path]) -> List[Note]:
        """Return all predicted notes without requiring tempo or writing files."""
        import soundfile as sf
        import torchaudio

        from .yourmt3.utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from .yourmt3.utils.note2event import mix_notes

        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if samples.shape[0] == 0 or not np.isfinite(samples).all():
            raise ValueError("Reference audio must contain finite, nonempty samples")
        audio = torch.from_numpy(samples.mean(axis=1))
        config = self.model.audio_cfg
        audio = torchaudio.functional.resample(
            audio, sample_rate, config["sample_rate"]
        )
        frames = config["input_frames"]
        padding = (-audio.numel()) % frames
        if padding:
            audio = torch.nn.functional.pad(audio, (0, padding))
        segments = audio.reshape(-1, 1, frames)
        predictions = []
        for batch in segments.split(8):
            with torch.autocast(
                self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
            ):
                predictions.append(
                    self.model.inference(batch.to(self.device)).cpu().numpy()
                )

        start_times = [
            index * frames / config["sample_rate"] for index in range(len(segments))
        ]
        channel_notes = []
        for channel in range(self.model.task_manager.num_decoding_channels):
            tokens = [batch[:, channel, :] for batch in predictions]
            events, _, _ = self.model.task_manager.detokenize_list_batches(
                tokens, start_times, return_events=True
            )
            notes, _ = merge_zipped_note_events_and_ties_to_notes(events)
            channel_notes.append(notes)
        return mix_notes(channel_notes)
