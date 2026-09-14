# YourMT3 inference components

Upstream project: [YourMT3](https://github.com/mimbres/YourMT3).
The [current upstream root license](https://github.com/mimbres/YourMT3/blob/main/LICENSE)
is GPL-3.0. The Apache-2.0 file headers described below must be reconciled with
the exact source revision and authorization before public distribution; no
verified source revision is recorded here yet.

The model, configuration, event codec, and note processing modules are adapted
from the YourMT3 inference source supplied for MuLaCover. Original source files
carry Copyright 2024 The YourMT3 Authors and Apache License 2.0 notices. See
[LICENSE-APACHE-2.0](LICENSE-APACHE-2.0). The associated paper is
https://arxiv.org/abs/2407.04822.

Modifications in MuLaCover: private package imports; a fixed inference-only
`torch.nn.Module` for the YPTF.MoE+Multi (noPS) checkpoint; explicit device and
checkpoint loading; removal of training, evaluation, UI, and logging entrypoints;
and retention of the original T5 tuple-cache implementation for compatibility
with MuLaCover's Transformers dependency.

`model/t5_compat.py` contains the necessary attention and feed-forward classes
from Hugging Face Transformers **4.45.1**,
`transformers/models/t5/modeling_t5.py`, distributed under Apache License 2.0.
The original Mesh TensorFlow, T5 Authors, and Hugging Face copyright notice is
preserved in that file.

`model/RoPE/RoPE.py` identifies its derivation from Phil Wang's
https://github.com/lucidrains/rotary-embedding-torch. Its source attribution is
preserved. See the accompanying rotary embedding license.

No pretrained weights are included in the source distribution. This notice
describes source-code provenance and does not grant additional rights to
separately supplied model weights.
