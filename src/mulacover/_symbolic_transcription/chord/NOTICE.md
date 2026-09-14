# Third-party chord inference code

Upstream project: [Music X Lab](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition).
See the [upstream license](https://github.com/music-x-lab/ISMIR2019-Large-Vocabulary-Chord-Recognition/blob/master/LICENSE).

This directory adapts Music X Lab's **Large-Vocabulary Chord Transcription via
Chord Structure Decomposition** (ISMIR 2019), distributed under the MIT license
in `LICENSE`. Copyright (c) 2023 Music X Lab.

The adapted sources are `chordnet_ismir_naive.py`, `complex_chord.py`,
`extractors/xhmm_ismir.py`, `extractors/cqt.py`, `chord_recognition.py`, and the
bundled chord vocabulary. The reference implementation is
the `ISMIR2019-Large-Vocabulary-Chord-Recognition` source supplied for MuLaCover.

Local changes retain only the inference network, vocabulary parsing, CQT
preprocessing, five-model probability averaging, and unrestricted Viterbi
decoding. Training and dataset code, optimizer loading, process-global models,
and MIDI conversion are removed. Imports are package-relative and deprecated
NumPy integer aliases are replaced. Chord labels retain extensions and inversions;
their interval endpoints use the reference implementation's integer-beat rounding.

Place the five original checkpoint files directly in the configured chord
checkpoint directory:

```text
joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s0.best.sdict
joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s1.best.sdict
joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s2.best.sdict
joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s3.best.sdict
joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s4.best.sdict
```

Inference requires PyTorch, NumPy, and librosa. No training framework, dataset
files, optimizer, or MIDI writer is required. Model weights are not bundled.

Float32 is the default and is required for CPU inference. CUDA inference also
accepts float16 and bfloat16; reduced precision may change decoded labels.
