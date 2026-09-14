# Contributing to MuLaCover

Thanks for helping improve MuLaCover. Before starting substantial work, open a
GitHub issue so the intended behavior and scope can be discussed.

## Development setup

```bash
conda create -y -n mulacover-dev python=3.10 pip
conda activate mulacover-dev
python -m pip install -e '.[audio]'
mulacover --help
```

## Pull requests

1. Keep changes focused and include reproducible validation steps for behavior changes.
2. Verify the affected inference path and include the command and results.
3. Update documentation when changing public APIs, inputs, or checkpoint layout.
4. Do not commit model weights, generated audio, private datasets, credentials,
   or media without documented redistribution permission.
5. Confirm that new dependencies and copied code have compatible licenses.

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 source-code license. This does not change the separate
terms covering official model weights and generated outputs in `MODEL_LICENSE`.

## Reporting problems

Use the bug-report template for reproducible defects and include environment,
GPU, CUDA, PyTorch, command, and logs. Do not post security vulnerabilities or
sensitive files publicly; follow `SECURITY.md` instead.
