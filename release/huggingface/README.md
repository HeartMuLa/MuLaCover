---
license: cc-by-nc-4.0
library_name: mulacover
pipeline_tag: audio-to-audio
tags:
  - music
  - audio
  - cover
  - cover-song-generation
  - music-remix
  - remix-generation
  - text-to-music
  - symbolic-music
---

# MuLaCover — Cover Song & Music Remix Generation

MuLaCover is a controllable AI cover-song and music-remix model. It transforms
reference music using lyrics and a structured text style prompt, or generates
from melody/chord MIDI. The standalone inference code is maintained in the
[MuLaCover repository](https://github.com/HeartMuLa/MuLaCover).

[HeartMuLa repository](https://github.com/HeartMuLa/heartlib) ·
[HeartMuLa paper](https://arxiv.org/abs/2601.10547)

> Model weights are publicly available for download. No access request is required.

## Inputs

Lyrics use section markers and preserved line breaks:

```text
[Verse]
One lyric line
Another lyric line

[Chorus]
One chorus line
```

Style is one UTF-8 line using named fields:

```text
topic:[Longing]; genre:[country]; instrument:[Strings,acoustic guitar]; mood:[hopeful]
```

## Download

Download the model weights with the Hugging Face CLI. Authentication is not required:

```bash
hf download HeartMuLa/MuLaCover --local-dir ./ckpt/MuLaCover
```

Also download HeartCodec and Qwen3-Embedding-0.6B
into sibling checkpoint directories. Reference-audio conditioning also needs
the documented YourMT3 and ChordNet checkpoints. See the code repository for
the complete installation and generation commands.

## Usage

Install the `mulacover` package and prepare all checkpoint directories using
the [generation guide](https://github.com/HeartMuLa/MuLaCover/blob/main/examples/cover_song_generation.md).
Run the following from the code repository root. `./ckpt` is the parent
directory containing MuLaCover, HeartCodec, Qwen3, and the transcription models.

```python
import torch
from mulacover import MuLaCoverGenPipeline

torch.manual_seed(42)
pipe = MuLaCoverGenPipeline.from_pretrained(
    "./ckpt",
    device=torch.device("cuda:0"),
    dtype={
        "mulacover": torch.bfloat16,
        "codec": torch.float32,
        "qwen": torch.float32,
        "transcriptor": torch.float32,
    },
    lazy_load=True,
)
pipe(
    {
        "ref_audio": "/path/to/reference.mp3",
        "lyrics": "assets/lyrics.txt",
        "tags": "assets/tags.txt",
    },
    save_path="cover.wav",
)
```

The pipeline writes a WAV file. Use the MuLaCover package API shown above;
this model is not a built-in Transformers model class.

## Files

- `config.json`: MuLaCover model architecture configuration.
- `gen_config.json`: audio/text token identifiers used during generation.
- `model-00001-of-00005.safetensors` … `model-00005-of-00005.safetensors`:
  sharded model weights.
- `model.safetensors.index.json`: shard index.
- `tokenizer.json`: lyrics and prompt tokenizer.
- `SHA256SUMS`: release integrity checksums.

## License and use restrictions

The official model weights are licensed under CC BY-NC 4.0 with the additional
terms in `MODEL_LICENSE`. Commercial use of the official weights is prohibited
without separate written authorization from MuLa Labs.

Outputs generated using the official weights are restricted to noncommercial
use unless separately authorized in writing by MuLa Labs. The Apache-2.0
license covering the source-code repository does not grant commercial rights
to these weights or their generated outputs.

Users are responsible for obtaining the rights required for their lyrics,
reference audio, performances, voices, and generated works. Do not use the
model to impersonate a person deceptively or violate copyright, privacy, or
publicity rights.

## Community

Join the [MuLaCover Discord](https://discord.gg/2Qj5DXsvh) for discussion and
community support. Reproducible software defects should be reported through
the code repository's GitHub Issues.

## Join MuLa Labs, Vera Praxis Lab

We are always excited to meet people with a strong interest in audio and music.
MuLa Labs has internship openings for candidates who want to build the next
generation of music and audio technology. To apply, email your resume or CV,
together with a short introduction, to
[contact@mulalabs.ai](mailto:contact@mulalabs.ai).
