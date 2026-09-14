# MuLaCover cover generation

Generate a cover from reference audio or melody/chord MIDI, plus lyrics and style
text. Run the following steps on a fresh Ubuntu 22.04 machine with an NVIDIA GPU
and a driver compatible with CUDA 13.0. The Python dependencies use PyTorch 2.10.0;
see [PyTorch installation options](https://pytorch.org/get-started/previous-versions/#v2100)
if your GPU requires a different CUDA build.

## Installation

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv git curl ffmpeg libsndfile1 gh

gh auth login --hostname github.com --git-protocol https
gh repo clone HeartMuLa/MuLaCover
cd MuLaCover
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[audio]'
python -c "import torch; from mulacover import MuLaCoverGenPipeline; assert torch.cuda.is_available(); print('Ready')"
```

Use a GitHub account with access to the `HeartMuLa/MuLaCover` repository.
Keep the virtual environment active and
run all remaining commands from the repository root.

## Download models

**MuLaCover and tokenizer:** authenticate with an account that has access to
the private release candidate, download it from Hugging Face, and verify it:

```bash
hf auth login
hf download HeartMuLa/MuLaCover --local-dir ckpt/MuLaCover
(cd ckpt/MuLaCover && sha256sum -c SHA256SUMS)
```

**Style encoder and audio codec:**

```bash
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ckpt/Qwen3-Embedding-0.6B
hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ckpt/HeartCodec-oss
```

**Audio transcription:** download the matching
[YourMT3 checkpoint](https://huggingface.co/spaces/mimbres/YourMT3/blob/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt)
and all five [ChordNet checkpoints](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/tree/master/cache_data).
Skip this step if you only use MIDI input.

```bash
mkdir -p ckpt/SymbolicTranscriptor/yourmt3 ckpt/SymbolicTranscriptor/chord
curl -fL \
  'https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt' \
  -o ckpt/SymbolicTranscriptor/yourmt3/last.ckpt
for fold in 0 1 2 3 4; do
  chord_file="joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s${fold}.best.sdict"
  curl -fL \
    "https://raw.githubusercontent.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/master/cache_data/$chord_file" \
    -o "ckpt/SymbolicTranscriptor/chord/$chord_file"
done
```

The final layout is:

```text
ckpt/
├── MuLaCover/                 # tokenizer, configs, five shards and index
├── HeartCodec-oss/
├── Qwen3-Embedding-0.6B/
└── SymbolicTranscriptor/
    ├── yourmt3/last.ckpt
    └── chord/                # five .best.sdict files
```

## Generate from audio

Use multiline lyrics with section markers on their own lines, one lyric line per
line, and a blank line between sections. Preserve these line breaks.
Use one-line named style fields, for example
`topic:[Longing]; genre:[country]; instrument:[Strings,acoustic guitar]; mood:[hopeful]`.

```bash
python examples/run_cover_song_generation.py \
  --model_path ./ckpt --ref_audio /path/to/reference.mp3 \
  --lyrics ./assets/lyrics.txt --tags ./assets/tags.txt \
  --symbolic_save_dir transcribed --save_path ./assets/cover.wav \
  --device cuda:0 --seed 42
```

`--symbolic_save_dir` saves melody, chord and drum MIDI for reuse. Use
`--bpm 120` to override automatic BPM estimation.

## Generate from MIDI

```bash
python examples/run_cover_song_generation.py \
  --model_path ./ckpt --melody_midi transcribed/melody.mid \
  --chord_midi transcribed/chord.mid --drum_midi transcribed/drums.mid \
  --lyrics ./assets/lyrics.txt --tags ./assets/tags.txt \
  --save_path ./assets/cover_from_midi.wav --device cuda:0 --seed 123
```

`--drum_midi` is optional. Audio and MIDI inputs cannot be combined. `--lyrics`
and `--tags` accept UTF-8 file paths or literal text. Preserve the multiline
lyrics and one-line named style fields shown above. Change `--seed` to sample a
different result. Lazy loading is enabled by default to reduce GPU memory use.

## Python API

```python
import torch
from mulacover import MuLaCoverGenPipeline

pipe = MuLaCoverGenPipeline.from_pretrained(
    "./ckpt", device=torch.device("cuda:0"),
    dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
           "qwen": torch.float32, "transcriptor": torch.float32},
    lazy_load=True,
)
pipe(
    {"ref_audio": "/path/to/reference.mp3",
     "lyrics": "./assets/lyrics.txt", "tags": "./assets/tags.txt"},
    save_path="./assets/cover.wav",
    temperature=1.0, topk=250, cfg_scale=1.5,
)
```

The call writes the audio file and returns `None`.
