"""Generate a cover from reference audio or melody/chord MIDI files."""

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from .configuration import MuLaCoverConfig
from .modeling import MuLaCover
from .symbolic import SymbolicCondition
from ._codec.modeling import HeartCodec


DEFAULT_MAX_AUDIO_LENGTH_MS = 300_000


@dataclass
class MuLaCoverGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0
    tag_start_id: int = 128021
    tag_end_id: int = 128022
    muq_start_id: int = 128023
    muq_end_id: int = 128024
    qwen_start_id: int = 128025
    qwen_end_id: int = 128026

    @classmethod
    def from_file(cls, path: Union[str, Path]):
        with open(path, encoding="utf-8") as fp:
            return cls(**json.load(fp))


def _resolve_paths(pretrained_path):
    root = Path(pretrained_path)
    model_root = root / "MuLaCover"
    if not model_root.is_dir():
        # Compatibility with private pre-release bundles created before the
        # public checkpoint directory was aligned with the product name.
        model_root = root / "MuLaCover-oss"
    tokenizer = root / "tokenizer.json"
    if not tokenizer.is_file():
        tokenizer = model_root / "tokenizer.json"
    paths = {
        "mulacover": model_root,
        "codec": root / "HeartCodec-oss",
        "qwen": root / "Qwen3-Embedding-0.6B",
        "transcriptor": root / "SymbolicTranscriptor",
        "tokenizer": tokenizer,
        "gen_config": model_root / "gen_config.json",
    }
    for name in ("mulacover", "codec", "qwen", "tokenizer", "gen_config"):
        path = paths[name]
        exists = (
            path.is_file() if name in ("tokenizer", "gen_config") else path.is_dir()
        )
        if not exists:
            raise FileNotFoundError(f"Missing {name} resource: {path}")
    return paths


def _resolve_components(value, kind):
    names = ("mulacover", "codec", "qwen", "transcriptor")
    if isinstance(value, dict):
        missing = set(names) - value.keys()
        if missing:
            raise ValueError(
                f"{kind} is missing components: {', '.join(sorted(missing))}"
            )
        values = {name: value[name] for name in names}
    else:
        values = dict.fromkeys(names, value)
    if kind == "device":
        return {name: torch.device(item) for name, item in values.items()}
    for name, item in values.items():
        if item not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported {kind} for {name}: {item}")
    return values


def _read_text(value, name):
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text or a text file path")
    # Long literal lyrics need not be valid filesystem names.
    try:
        if Path(value).is_file():
            return Path(value).read_text(encoding="utf-8")
    except OSError:
        pass
    return value


class MuLaCoverGenPipeline:
    """A single-song pipeline with audio and symbolic input modes.

    ``lyrics`` and ``tags`` accept text or text-file paths. Supply either
    ``ref_audio`` (optionally ``bpm``), or ``melody_midi`` and ``chord_midi``
    (optionally ``drum_midi``). Audio transcription dependencies and weights
    are only required when using ``ref_audio``.
    """

    def __init__(
        self,
        paths: Dict[str, Path],
        devices: Dict[str, torch.device],
        dtypes: Dict[str, torch.dtype],
        text_tokenizer: Tokenizer,
        config: MuLaCoverGenConfig,
        model_config: MuLaCoverConfig,
        lazy_load: bool = False,
    ):
        self.paths = paths
        self.devices = devices
        self.dtypes = dtypes
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.model_config = model_config
        self.lazy_load = lazy_load
        self._mulacover = None
        self._codec = None
        self._qwen = None
        self._qwen_tokenizer = None
        self._transcriptor = None
        self._cache_batch_size = None
        self.text_tokenizer.no_truncation()
        if not lazy_load:
            self.mulacover
            self.codec
            self._load_qwen()

    @property
    def mulacover(self):
        if self._mulacover is None:
            model, loading_info = MuLaCover.from_pretrained(
                self.paths["mulacover"],
                device_map=self.devices["mulacover"],
                dtype=self.dtypes["mulacover"],
                local_files_only=True,
                output_loading_info=True,
            )
            missing_style = [
                name for name in loading_info["missing_keys"]
                if name.startswith("tag_mlp_layers.")
            ]
            if model.config.train_tag and missing_style:
                raise ValueError(
                    "This MuLaCover checkpoint is missing StyleMLP weights. "
                    "Download a complete MuLaCover model release containing "
                    "the StyleMLP weights; incomplete checkpoints cannot "
                    "reproduce the intended style conditioning."
                )
            self._mulacover = model.eval()
        return self._mulacover

    @property
    def codec(self):
        if self._codec is None:
            self._codec = HeartCodec.from_pretrained(
                self.paths["codec"],
                device_map=self.devices["codec"],
                dtype=self.dtypes["codec"],
                local_files_only=True,
            ).eval()
        return self._codec

    def _load_qwen(self):
        if self._qwen is None:
            from transformers import AutoModel, AutoTokenizer

            self._qwen_tokenizer = AutoTokenizer.from_pretrained(
                self.paths["qwen"], padding_side="left", local_files_only=True
            )
            self._qwen = (
                AutoModel.from_pretrained(
                    self.paths["qwen"],
                    dtype=self.dtypes["qwen"],
                    local_files_only=True,
                )
                .to(self.devices["qwen"])
                .eval()
            )

    def _release(self, name):
        if not self.lazy_load:
            return
        setattr(self, f"_{name}", None)
        if name == "qwen":
            self._qwen_tokenizer = None
        if name == "mulacover":
            self._cache_batch_size = None
        gc.collect()
        device = self.devices[name]
        if device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    def _sanitize_parameters(self, **kwargs):
        allowed = {
            "cfg_scale",
            "max_audio_length_ms",
            "temperature",
            "topk",
            "save_path",
            "symbolic_save_dir",
            "disable_progress",
        }
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(f"Unexpected parameters: {', '.join(sorted(unexpected))}")
        cfg_scale = kwargs.get("cfg_scale", 1.5)
        temperature = kwargs.get("temperature", 1.0)
        topk = kwargs.get("topk", 250)
        duration = kwargs.get("max_audio_length_ms", DEFAULT_MAX_AUDIO_LENGTH_MS)
        for name, value in (("cfg_scale", cfg_scale), ("temperature", temperature)):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(topk, bool)
            or not isinstance(topk, int)
            or not 1 <= topk <= self.model_config.audio_vocab_size
        ):
            raise ValueError("topk must be between 1 and the audio vocabulary size")
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 80:
            raise ValueError("max_audio_length_ms must be an integer of at least 80")
        return (
            {
                "cfg_scale": cfg_scale,
                "max_audio_length_ms": duration,
                "symbolic_save_dir": kwargs.get("symbolic_save_dir"),
            },
            {
                "cfg_scale": cfg_scale,
                "max_audio_length_ms": duration,
                "temperature": temperature,
                "topk": topk,
                "disable_progress": kwargs.get("disable_progress", False),
            },
            {
                "save_path": kwargs.get("save_path", "output.wav"),
                "disable_progress": kwargs.get("disable_progress", False),
            },
        )

    def _text_ids(self, text):
        return self.text_tokenizer.encode(text, add_special_tokens=False).ids

    def _build_prompt(self, lyrics, tags):
        c = self.config
        lyric_ids = [c.text_bos_id] + self._text_ids(lyrics) + [c.text_eos_id]
        tag_ids = [c.tag_start_id] + self._text_ids(tags) + [c.tag_end_id]
        qwen_index, muq_index = len(tag_ids) + 1, len(tag_ids) + 4
        ids = (
            tag_ids
            + [c.qwen_start_id, 0, c.qwen_end_id, c.muq_start_id, 0, c.muq_end_id]
            + lyric_ids
        )
        tokens = torch.full(
            (len(ids), self.model_config.audio_num_codebooks + 1),
            c.empty_id,
            dtype=torch.long,
        )
        tokens[:, -1] = torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask[:, -1] = True
        return tokens, mask, qwen_index, muq_index

    @torch.inference_mode()
    def _encode_style(self, tags):
        if not tags:
            return torch.zeros(self.model_config.qwen_dim)
        try:
            self._load_qwen()
            batch = self._qwen_tokenizer(
                [tags], padding=True, truncation=False, return_tensors="pt"
            )
            if batch["input_ids"].shape[1] > 8192:
                raise ValueError("Style exceeds the Qwen limit of 8192 tokens")
            batch = {
                key: value.to(self.devices["qwen"]) for key, value in batch.items()
            }
            hidden = self._qwen(**batch).last_hidden_state
            # Left padding makes the final position the last real token.
            embedding = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
            if embedding.shape[-1] != self.model_config.qwen_dim:
                raise ValueError("The style encoder dimension does not match MuLaCover")
            return embedding[0].cpu()
        finally:
            self._release("qwen")

    def _symbolic_condition(self, inputs):
        audio = inputs.get("ref_audio")
        midi_keys = ("melody_midi", "chord_midi", "drum_midi")
        if audio is not None:
            if any(inputs.get(key) is not None for key in midi_keys):
                raise ValueError("ref_audio and MIDI inputs are mutually exclusive")
            if not Path(audio).is_file():
                raise FileNotFoundError(f"Reference audio not found: {audio}")
            from ._symbolic_transcription import SymbolicTranscriber

            try:
                if self._transcriptor is None:
                    self._transcriptor = SymbolicTranscriber(
                        self.paths["transcriptor"],
                        self.devices["transcriptor"],
                        self.dtypes["transcriptor"],
                        lazy_load=self.lazy_load,
                    )
                return self._transcriptor.transcribe(audio, bpm=inputs.get("bpm"))
            finally:
                self._release("transcriptor")
        if inputs.get("melody_midi") is None or inputs.get("chord_midi") is None:
            raise ValueError("Provide ref_audio, or both melody_midi and chord_midi")
        if inputs.get("bpm") is not None:
            raise ValueError(
                "bpm is only used with ref_audio; MIDI uses its musical ticks"
            )
        return SymbolicCondition.from_midi(
            inputs["melody_midi"], inputs["chord_midi"], inputs.get("drum_midi")
        )

    def preprocess(
        self,
        inputs: Dict[str, Any],
        cfg_scale=1.5,
        max_audio_length_ms=DEFAULT_MAX_AUDIO_LENGTH_MS,
        symbolic_save_dir=None,
    ):
        lyrics = _read_text(inputs["lyrics"], "lyrics").lower()
        tags = _read_text(inputs["tags"], "tags").strip().lower()
        if not lyrics.strip():
            raise ValueError("lyrics must not be empty")
        if tags.startswith("<tag>") and tags.endswith("</tag>"):
            tags = tags[len("<tag>") : -len("</tag>")].strip()
        tokens, mask, qwen_index, muq_index = self._build_prompt(lyrics, tags)
        if len(tokens) + max_audio_length_ms // 80 > 8192:
            raise ValueError(
                "The prompt and requested audio exceed the 8192-token context"
            )
        condition = self._symbolic_condition(inputs)
        if symbolic_save_dir is not None:
            condition.save_midi(symbolic_save_dir)
        symbolic = condition.to_tensors()
        style = self._encode_style(tags)
        batch_size = 2 if cfg_scale > 1 else 1

        def batch(tensor):
            return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim))

        return {
            "tokens": batch(tokens),
            "tokens_mask": batch(mask),
            "input_pos": batch(torch.arange(len(tokens), dtype=torch.long)),
            "qwen_embedding": batch(style),
            "muq_embedding": batch(torch.zeros(self.model_config.muq_dim)),
            "qwen_indices": torch.full((batch_size,), qwen_index, dtype=torch.long),
            "muq_indices": torch.full((batch_size,), muq_index, dtype=torch.long),
            **{key: batch(value) for key, value in symbolic.items()},
        }

    @torch.inference_mode()
    def _forward(
        self,
        model_inputs,
        max_audio_length_ms=DEFAULT_MAX_AUDIO_LENGTH_MS,
        temperature=1.0,
        topk=250,
        cfg_scale=1.5,
        disable_progress=False,
    ):
        device, dtype = self.devices["mulacover"], self.dtypes["mulacover"]
        inputs = {key: value.to(device) for key, value in model_inputs.items()}
        for key in ("pianoroll", "drum_pianoroll", "chord"):
            inputs[key] = inputs[key].to(dtype)
        tokens = inputs.pop("tokens")
        mask = inputs.pop("tokens_mask")
        position = inputs.pop("input_pos")
        batch_size = tokens.shape[0]
        frames = []
        try:
            model = self.mulacover
            if self._cache_batch_size != batch_size:
                self._cache_batch_size = None
                model.setup_caches(batch_size)
                self._cache_batch_size = batch_size
            model.reset_caches()
            autocast = (
                torch.autocast(device.type, dtype=dtype)
                if dtype != torch.float32
                else nullcontext()
            )
            with autocast:
                for step in tqdm(
                    range(max_audio_length_ms // 80), disable=disable_progress
                ):
                    sample = model.generate_frame(
                        tokens=tokens,
                        tokens_mask=mask,
                        input_pos=position,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        first_step=step == 0,
                        **inputs,
                    )
                    if torch.any(sample[0] >= self.config.audio_eos_id):
                        break
                    if torch.any(sample[0] < 0) or torch.any(
                        sample[0] >= self.config.audio_eos_id - 1
                    ):
                        raise RuntimeError("MuLaCover generated an audio padding token")
                    frames.append(sample[0].cpu())
                    tokens = torch.full(
                        (batch_size, 1, self.model_config.audio_num_codebooks + 1),
                        self.config.empty_id,
                        dtype=torch.long,
                        device=device,
                    )
                    tokens[:, 0, :-1] = sample
                    mask = torch.ones_like(tokens, dtype=torch.bool)
                    mask[..., -1] = False
                    position = position[:, -1:] + 1
                    inputs["qwen_indices"] = None
                    inputs["muq_indices"] = None
                    inputs["muq_embedding"] = None
            if not frames:
                raise RuntimeError("MuLaCover emitted EOS before generating any audio")
            return {"frames": torch.stack(frames, dim=-1)}
        finally:
            try:
                if self._mulacover is not None and self._cache_batch_size is not None:
                    self._mulacover.reset_caches()
            finally:
                # Drop the local reference before releasing GPU storage.
                model = None
                self._release("mulacover")

    @torch.inference_mode()
    def postprocess(self, model_outputs, save_path, disable_progress=False):
        import soundfile as sf

        try:
            codec = self.codec
            waveform = codec.detokenize(
                model_outputs["frames"].to(self.devices["codec"]),
                duration=29.76,
                num_steps=10,
                guidance_scale=1.25,
                disable_progress=disable_progress,
            )
            sample_rate = codec.sample_rate
            waveform = waveform.detach().float().cpu()
        finally:
            codec = None
            self._release("codec")
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, waveform.numpy().T, sample_rate)

    @torch.inference_mode()
    def __call__(self, inputs: Dict[str, Any], **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = (
            self._sanitize_parameters(**kwargs)
        )
        model_inputs = self.preprocess(inputs, **preprocess_kwargs)
        model_outputs = self._forward(model_inputs, **forward_kwargs)
        self.postprocess(model_outputs, **postprocess_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: Union[torch.device, Dict[str, torch.device]],
        dtype: Union[torch.dtype, Dict[str, torch.dtype]],
        lazy_load: bool = False,
    ):
        paths = _resolve_paths(pretrained_path)
        return cls(
            paths=paths,
            devices=_resolve_components(device, "device"),
            dtypes=_resolve_components(dtype, "dtype"),
            text_tokenizer=Tokenizer.from_file(str(paths["tokenizer"])),
            config=MuLaCoverGenConfig.from_file(paths["gen_config"]),
            model_config=MuLaCoverConfig.from_pretrained(
                paths["mulacover"], local_files_only=True
            ),
            lazy_load=lazy_load,
        )
